import asyncio

from django.core.management.base import BaseCommand
from django.db.models import Q

from embeddings.service import embedding_service
from items.models import Item
from items.tasks import generate_item_embedding


class Command(BaseCommand):
    help = "Scan all un-embedded items and do embedding."

    def handle(self, *args, **options):
        asyncio.run(self.a_handle(*args, **options))

    async def a_handle(self, *args, **options):
        self.stdout.write("Starting to embed items...")

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
            generate_item_embedding.delay(item_id, "search_document")
        self.stdout.write(f"Queued {count} items for embedding.")

        self.stdout.write(self.style.SUCCESS("Finished embedding items."))
