import logging
from functools import reduce
from operator import add

from django.conf import settings
from django.contrib.postgres.search import SearchVector
from pgvector.django import CosineDistance

from embeddings.service import embedding_service
from items.models import Item, ItemEmbedding

logger = logging.getLogger(__name__)


async def engine_search(query, top_k, use_fts: bool = False) -> list[dict[str, ...]]:
    # Step 1: Generate or fetch cached query embedding
    try:
        query_vector = await embedding_service.aget_or_create_query_embedding(query)

        # Step 2: Run async ORM query
        # Optional hybrid FTS:
        # First, define the subquery for full-text search without awaiting it.
        if use_fts:
            vectors = (
                SearchVector("title", "description", config=lang.strip()) for lang in settings.FULLTEXT_SEARCH_LANGUAGES
            )
            combined_vector = reduce(add, vectors)
            fts_qs = Item.objects.annotate(search=combined_vector).filter(search=query)
            base_qs = ItemEmbedding.objects.filter(item__in=fts_qs)
        else:
            base_qs = ItemEmbedding.objects

        # Annotate with similarity and fetch results
        results_qs = (
            base_qs.annotate(distance=CosineDistance("vector", query_vector))
            .order_by("distance")
            .select_related("item")[:top_k]
        )

        # Construct the response payload with item data and similarity score
        # Ninja expects a flat dictionary that matches the SearchResultSchema.
        items: list[dict[str, ...]] = [
            {
                **emb.item.__dict__,  # Unpack the item's fields
                "distance": round(emb.distance, 4),
            }
            async for emb in results_qs
        ]
        return items
    except Exception as e:
        logger.error(f"Error in engine_search: {e}")
        raise
