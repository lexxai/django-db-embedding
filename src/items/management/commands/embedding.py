import asyncio

from django.core.management.base import BaseCommand

from items.embeddings import create_item_embedding
from items.models import Item


class Command(BaseCommand):
    help = "Scan all un-embedded items and do embedding."

    def handle(self, *args, **options):
        asyncio.run(self.a_handle(*args, **options))

    async def a_handle(self, *args, **options):
        self.stdout.write("Starting to embed items...")
        items_to_embed = Item.objects.filter(embedding__isnull=True)
        count = await items_to_embed.acount()
        self.stdout.write(f"Found {count} items to embed.")

        async for item in items_to_embed:
            self.stdout.write(f"Embedding item {item.id}: {item.title}")
            await create_item_embedding(item)

        self.stdout.write(self.style.SUCCESS("Finished embedding items."))
