import logging
from functools import reduce
from operator import add

from asgiref.sync import async_to_sync
from celery import shared_task
from django.conf import settings
from django.contrib.postgres.search import SearchVector
from django.core.exceptions import ValidationError

from embeddings.service import embedding_service
from .models import Item, ItemEmbedding

logger = logging.getLogger(__name__)


async def _generate_item_embedding_async(item_id: int, input_type: str = None):
    try:
        if not embedding_service:
            logger.error("Embedding service not initialized")
            return
        model_name = embedding_service.backend.model_name
        item = await Item.objects.aget(id=item_id)
        text = f"{item.title} {item.description}"
        # print(f"_generate_item_embedding_async text: {text}")
        vector = await embedding_service.aget_or_create_query_embedding(text, input_type)
        if vector is None or len(vector) != settings.VECTOR_EMBEDDIG_DIMENSIONS:
            raise ValidationError(f"Invalid vector length from embedding service. {item.title=}")

        await ItemEmbedding.objects.aupdate_or_create(
            item=item,
            defaults={"vector": vector, "model": model_name},
        )
        logger.info(f"Successfully generated embedding for item_id={item_id}")
    except Item.DoesNotExist:
        logger.warning(f"Item with id={item_id} does not exist. Task will not be retried.")
        # Do not re-raise, as this is not a failure worth retrying.


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_item_embedding(self, item_id: int, input_type: str = None):
    """
    Synchronous Celery task wrapper for the async embedding generation logic.
    """
    try:
        # Use asyncio.run() to execute the async function in a sync context.
        async_to_sync(_generate_item_embedding_async)(item_id, input_type)
    except ValidationError as e:
        # Catch non-retriable errors here. Log and do not re-raise.
        # This marks the task as FAILED but prevents Celery from retrying it.
        logger.error(f"Task failed permanently for item_id={item_id} due to validation error: {e}")
        # We need to explicitly mark the task as failed if we catch the exception.
        # The state will be FAILURE, but autoretry won't trigger.
        self.update_state(state="FAILURE", meta={"exc_type": type(e).__name__, "exc_message": str(e)})
        # We raise Ignore to tell Celery to stop processing and not retry.
        # from celery.exceptions import Ignore
        # raise Ignore()
        raise


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def update_item_vector_search(self, item_id: int):
    """
    Updates the pre-computed search vector for a given item, supporting multiple languages.
    """
    # Combine SearchVectors for multiple languages using the same logic as the repository.
    vectors = (SearchVector("title", "description", config=lang.strip()) for lang in settings.FULLTEXT_SEARCH_LANGUAGES)
    combined_vector = reduce(add, vectors)

    # Use .update() for an efficient, single SQL query without loading the object.
    rows_updated = Item.objects.filter(id=item_id).update(search_vector=combined_vector)

    if rows_updated == 0:
        logger.warning(f"Item with id={item_id} not found for search vector update. Task will not be retried.")
    else:
        logger.info(f"Successfully updated search vector for item_id={item_id}")
