import asyncio

from django.core.management.base import BaseCommand
from django.db.models import Q

from embeddings.service import embedding_service
from items.models import Item
from items.tasks import (
    generate_item_embedding,
    generate_item_embedding_in_batch,
    _generate_item_embedding_async,
    agenerate_item_embedding_in_batch,
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
            "--use-queue",
            action="store_true",
            default=False,
            help="Enable to use background queue for batch processing. Default: Disabled",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        if batch_size is None:
            asyncio.run(self.a_handle(*args, **options))
        else:
            asyncio.run(self.a_handle_batch(*args, **options))

    async def a_handle(self, *args, **options):

        use_queue = options["use_queue"]
        self.stdout.write(f"Starting to embed items... {use_queue=}")

        if not embedding_service:
            self.stdout.write(self.style.ERROR("Embedding service not initialized"))
            return None

        model_name = embedding_service.backend.model_name

        # Use ~Q for "not equal" instead of the non-existent `__ne` lookup.
        # Also, prefetch the IDs to avoid iterating over a large queryset.
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

    async def a_handle_batch(self, *args, **options):
        batch_size = options["batch_size"]
        use_queue = options["use_queue"]
        self.stdout.write(f"Starting to embed items... {use_queue=}")
        if use_queue:
            generate_item_embedding_in_batch.delay(batch_size, embedding_service.backend.InputType.DOCUMENT)
        else:
            await agenerate_item_embedding_in_batch(batch_size, embedding_service.backend.InputType.DOCUMENT)
        self.stdout.write(self.style.SUCCESS("Finished embedding items."))
