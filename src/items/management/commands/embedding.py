import asyncio

from django.conf import settings
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
        free_tier: bool = settings.OPENAI_API_FREE_TIER
        free_tier_delay: float = 60 / (settings.OPENAI_API_DELAY_TIME_RPM or 1)
        i = 0
        async for item in items_to_embed:
            i += 1
            self.stdout.write(f"Embedding item {item.id}: {item.title}")
            try:
                result = await create_item_embedding(item)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Failed to embed item {item.id}: {e}"))
                result = None
            if result is None:
                self.stdout.write(self.style.ERROR(f"Failed to embed item {item.id}"))
            if free_tier and i < count:
                self.stdout.write(
                    f"Sleep for free_tier delay: {free_tier_delay:.2} sec. ({settings.OPENAI_API_DELAY_TIME_RPM} RPM)"
                )
                await asyncio.sleep(free_tier_delay)

        self.stdout.write(self.style.SUCCESS("Finished embedding items."))
