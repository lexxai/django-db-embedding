import logging

from django.conf import settings

from embeddings.service import embedding_service
from .models import Item, ItemEmbedding

logger = logging.getLogger(__name__)


async def create_item_embedding(item: "Item"):
    """
    Creates and saves an embedding for a given item.
    If an embedding already exists for the item, it will be updated.
    """
    if embedding_service is None:
        logger.error("Embedding service is not initialized")
        return None

    text = f"{item.title}\n{item.description}"
    vector = await embedding_service.aget_or_create_query_embedding(text)

    embedding, created = await ItemEmbedding.objects.aupdate_or_create(
        item=item,
        defaults={"vector": vector, "model": settings.EMBEDDIG_MODEL_NAME},
    )
    return embedding
