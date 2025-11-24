import asyncio

from django.core.management.base import BaseCommand
from django.db.models import Q

from embeddings.service import embedding_service
from items.models import Item
from items.tasks import (
    generate_item_embedding,
    generate_item_embedding_in_batch,
    _generate_item_embedding_async,
    _agenerate_item_embedding_in_batch,
    _generate_item_embedding_in_batch,
    _generate_item_embedding,
)


class Command(BaseCommand):
    help = "Scan all un-embedded items and do embedding."

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of items to process in a single batch., Default 100",
        )
        parser.add_argument(
            "--queue",
            action="store_true",
            default=False,
            help="Enable to use background queue for batch processing. Default: Disabled",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            default=False,
            help="Allow overwriting existing embeddings. Default: Disabled",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        if not embedding_service:
            self.stdout.write(self.style.ERROR("Embedding service not initialized"))
            return None
        if embedding_service.is_async_prefer:
            if batch_size is None:
                asyncio.run(self._a_handle(*args, **options))
            else:
                asyncio.run(self._a_handle_batch(*args, **options))
        else:
            if batch_size is None:
                self._handle(*args, **options)
            else:
                self._handle_batch(*args, **options)
        return None

    async def _a_handle(self, *args, **options):

        use_queue = options["queue"]
        use_overwrite = options["overwrite"]
        self.stdout.write(f"Starting to embed items... {use_queue=}, {use_overwrite=}")

        if not embedding_service:
            self.stdout.write(self.style.ERROR("Embedding service not initialized"))
            return None

        model_name = embedding_service.backend.model_name

        # Use ~Q for "not equal" instead of the non-existent `__ne` lookup.
        # Also, prefetch the IDs to avoid iterating over a large queryset.
        if use_overwrite:
            items_to_embed_qs = (
                Item.objects.filter(embedding__model=model_name).order_by("id").values_list("id", flat=True)
            )
        else:
            items_to_embed_qs = (
                Item.objects.filter(Q(embedding__isnull=True) | ~Q(embedding__model=model_name))
                .order_by("id")
                .values_list("id", flat=True)
            )
        item_ids = [item_id async for item_id in items_to_embed_qs]
        count = len(item_ids)

        self.stdout.write(f"Found {count} items to embed.")

        if count == 0:
            self.stdout.write(self.style.SUCCESS("All items are already embedded with the current model."))
            return

        for item_id in item_ids:
            if use_queue:
                generate_item_embedding.delay(item_id, embedding_service.backend.InputType.DOCUMENT)
            else:
                await _generate_item_embedding_async(item_id, embedding_service.backend.InputType.DOCUMENT)

        self.stdout.write(f"Queued {count} items for embedding.")

        self.stdout.write(self.style.SUCCESS("Finished embedding items."))

    def _handle(self, *args, **options):
        use_queue = options["queue"]
        use_overwrite = options["overwrite"]
        self.stdout.write(f"Starting to embed items... {use_queue=}, {use_overwrite=}")

        if not embedding_service:
            self.stdout.write(self.style.ERROR("Embedding service not initialized"))
            return None

        model_name = embedding_service.backend.model_name

        # Use ~Q for "not equal" instead of the non-existent `__ne` lookup.
        # Also, prefetch the IDs to avoid iterating over a large queryset.
        if use_overwrite:
            items_to_embed_qs = (
                Item.objects.filter(embedding__model=model_name).order_by("id").values_list("id", flat=True)
            )
        else:
            items_to_embed_qs = (
                Item.objects.filter(Q(embedding__isnull=True) | ~Q(embedding__model=model_name))
                .order_by("id")
                .values_list("id", flat=True)
            )

        item_ids = [item_id for item_id in items_to_embed_qs]
        count = len(item_ids)

        self.stdout.write(f"Found {count} items to embed.")

        if count == 0:
            self.stdout.write(self.style.SUCCESS("All items are already embedded with the current model."))
            return

        for item_id in item_ids:
            if use_queue:
                generate_item_embedding.delay(item_id, embedding_service.backend.InputType.DOCUMENT)
            else:
                _generate_item_embedding(item_id, embedding_service.backend.InputType.DOCUMENT)

        self.stdout.write(f"Queued {count} items for embedding.")
        self.stdout.write(self.style.SUCCESS("Finished embedding items."))
        return None

    async def _a_handle_batch(self, *args, **options):
        batch_size = options["batch_size"]
        use_queue = options["queue"]
        use_overwrite = options["overwrite"]
        self.stdout.write(f"Starting batching to embed items... {use_queue=}, {use_overwrite=}")
        if use_queue:
            # Celery only supports sync code task only
            generate_item_embedding_in_batch.delay(
                batch_size, embedding_service.backend.InputType.DOCUMENT, overwrite=use_overwrite
            )
        else:
            await _agenerate_item_embedding_in_batch(
                batch_size, embedding_service.backend.InputType.DOCUMENT, overwrite=use_overwrite
            )
        self.stdout.write(self.style.SUCCESS("Finished embedding items."))

    def _handle_batch(self, *args, **options):
        batch_size = options["batch_size"]
        use_queue = options["queue"]
        use_overwrite = options["overwrite"]
        self.stdout.write(f"Starting to embed items... {use_queue=}, {use_overwrite=}")
        if use_queue:
            # Celery only supports sync code task only
            generate_item_embedding_in_batch.delay(
                batch_size, embedding_service.backend.InputType.DOCUMENT, overwrite=use_overwrite
            )
        else:
            _generate_item_embedding_in_batch(
                batch_size, embedding_service.backend.InputType.DOCUMENT, overwrite=use_overwrite
            )
        self.stdout.write(self.style.SUCCESS("Finished embedding items."))
        return None
