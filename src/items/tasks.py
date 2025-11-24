import logging
from functools import reduce
from operator import add

from asgiref.sync import async_to_sync
from celery import shared_task
from django.conf import settings
from celery.exceptions import Ignore
from django.contrib.postgres.search import SearchVector
from django.core.exceptions import ValidationError
from django.db.models import Q

from embeddings.embedding_backend import EmbeddingBackend
from embeddings.service import embedding_service
from .models import Item, ItemEmbedding
from .repository import acreate_item_embeddings_in_batch, create_item_embeddings_in_batch

logger = logging.getLogger(__name__)


async def _generate_item_embedding_async(item_id: int, input_type: EmbeddingBackend.InputType = None):
    try:
        if not embedding_service:
            logger.error("Embedding service not initialized")
            return
        model_name = embedding_service.backend.model_name
        item = await Item.objects.aget(id=item_id)
        text = f"{item.title} {item.description}"
        # print(f"_generate_item_embedding_async text: {text}")
        vector = await embedding_service.aget_or_create_query_embedding(text, input_type)
        if vector is None or len(vector) > settings.VECTOR_EMBEDDING_DIMENSIONS:
            raise ValidationError(f"Invalid vector length from embedding service. {item.title=}")

        await ItemEmbedding.objects.aupdate_or_create(
            item=item,
            defaults={"vector": vector, "model": model_name},
        )
        logger.info(f"Successfully generated embedding for item_id={item_id}")
    except Item.DoesNotExist:
        logger.warning(f"Item with id={item_id} does not exist. Task will not be retried.")
        # Do not re-raise, as this is not a failure worth retrying.


def _generate_item_embedding(item_id: int, input_type: EmbeddingBackend.InputType = None):
    try:
        if not embedding_service:
            logger.error("Embedding service not initialized")
            return
        model_name = embedding_service.backend.model_name
        item = Item.objects.get(id=item_id)
        text = f"{item.title} {item.description}"
        # print(f"_generate_item_embedding_async text: {text}")
        vector = embedding_service.get_or_create_query_embedding(text, input_type)
        if vector is None or len(vector) > settings.VECTOR_EMBEDDING_DIMENSIONS:
            raise ValidationError(f"Invalid vector length from embedding service. {item.title=}")

        ItemEmbedding.objects.update_or_create(
            item=item,
            defaults={"vector": vector, "model": model_name},
        )
        logger.info(f"Successfully generated embedding for item_id={item_id}")
    except Item.DoesNotExist:
        logger.warning(f"Item with id={item_id} does not exist. Task will not be retried.")
        # Do not re-raise, as this is not a failure worth retrying.


def _generate_item_embedding_in_batch(
    batch_size: int, input_type: EmbeddingBackend.InputType = None, overwrite: bool = False
):
    logger.debug(f"Starting to embed items... {input_type=}")
    if not embedding_service:
        logger.error("Embedding service not initialized")
        return

    model_name = embedding_service.backend.model_name

    if overwrite:
        items_to_embed_qs = Item.objects.filter(embedding__model=model_name)
    else:
        items_to_embed_qs = Item.objects.filter(Q(embedding__isnull=True) | ~Q(embedding__model=model_name))
    total_count = items_to_embed_qs.count()
    logger.debug(f"Found {total_count} items to embed.")

    # Process in batches
    for i in range(0, total_count, batch_size):
        # `await` on a slice already returns a list, so `list()` is not needed.
        # Use .values() to fetch only the necessary data, which is much more memory-efficient.
        batch_data = [
            item_dict for item_dict in items_to_embed_qs.values("id", "title", "description")[i : i + batch_size]
        ]
        if not batch_data:
            break

        logger.debug(f"Processing batch of {len(batch_data)} items...")
        create_item_embeddings_in_batch(batch_data, input_type)

    logger.debug("Finished embedding items.")


async def _agenerate_item_embedding_in_batch(
    batch_size: int, input_type: EmbeddingBackend.InputType = None, overwrite: bool = False
):
    logger.debug("Async Starting to embed items...  {input_type=}")
    if not embedding_service:
        logger.error("Embedding service not initialized")
        return

    model_name = embedding_service.backend.model_name
    if overwrite:
        items_to_embed_qs = Item.objects.filter(embedding__model=model_name)
    else:
        items_to_embed_qs = Item.objects.filter(Q(embedding__isnull=True) | ~Q(embedding__model=model_name))
    total_count = await items_to_embed_qs.acount()
    logger.debug(f"Found {total_count} items to embed.")

    # Process in batches
    for i in range(0, total_count, batch_size):
        # `await` on a slice already returns a list, so `list()` is not needed.
        # Use .values() to fetch only the necessary data, which is much more memory-efficient.
        batch_data = [
            item_dict async for item_dict in items_to_embed_qs.values("id", "title", "description")[i : i + batch_size]
        ]
        if not batch_data:
            break

        logger.debug(f"Processing batch of {len(batch_data)} items...")
        await acreate_item_embeddings_in_batch(batch_data, input_type)

    logger.debug("Finished embedding items.")


# TASKS


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_item_embedding(self, item_id: int, input_type: EmbeddingBackend.InputType = None):
    """
    Synchronous Celery task wrapper for the async embedding generation logic.
    """
    if not embedding_service:
        logger.error("Embedding service not initialized")
        return
    try:
        # Use asyncio.run() to execute the async function in a sync context.
        if embedding_service.is_async_prefer:
            async_to_sync(_generate_item_embedding_async)(item_id, input_type)
        else:
            _generate_item_embedding(item_id, input_type)
    except ValidationError as e:
        # Catch non-retriable errors here. Log and do not re-raise.
        # This marks the task as FAILED but prevents Celery from retrying it.
        logger.error(f"Task failed permanently for item_id={item_id} due to validation error: {e}")
        # We need to explicitly mark the task as failed if we catch the exception.
        # The state will be FAILURE, but autoretry won't trigger.
        self.update_state(state="FAILURE", meta={"exc_type": type(e).__name__, "exc_message": str(e)})
        # We raise Ignore to tell Celery to stop processing and not retry.
        # Re-raising the original exception would trigger autoretry.
        raise Ignore()


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


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def generate_item_embedding_in_batch(self, batch_size: int = None, input_type: EmbeddingBackend.InputType = None):
    # Use a default batch size from settings if not provided to prevent errors.
    final_batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE or 100
    if not embedding_service:
        logger.error("Embedding service not initialized")
        return
    if embedding_service.is_async_prefer:
        async_to_sync(_agenerate_item_embedding_in_batch)(final_batch_size, input_type)
    else:
        _generate_item_embedding_in_batch(final_batch_size, input_type)
