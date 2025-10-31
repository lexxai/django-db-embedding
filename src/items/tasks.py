import logging

from celery import shared_task
from django.conf import settings
from ninja.errors import ValidationError

from embeddings.service import embedding_service
from .models import Item, ItemEmbedding

logger = logging.getLogger(__name__)


@shared_task
async def generate_item_embedding(item_id):
    try:
        item = await Item.objects.aget(id=item_id)
        text = f"{item.title} {item.description}"
        vector = await embedding_service.aget_or_create_query_embedding(text)
        if len(vector) != settings.VECTOR_EMBEDDIG_DIMENSIONS:
            raise ValidationError([{"error": "Invalid vector length"}])

        await ItemEmbedding.objects.aupdate_or_create(
            item=item,
            defaults={"vector": vector, "model": settings.EMBEDDIG_MODEL_NAME},
        )
    except (Item.DoesNotExist, ValidationError, Exception) as e:
        logger.error(str(e).split("\n")[0])
