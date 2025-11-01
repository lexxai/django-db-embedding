import logging
from functools import reduce
from operator import or_

from django.conf import settings
from django.contrib.postgres.search import SearchRank, SearchQuery
from django.db.models import F, Q
from pgvector.django import CosineDistance

from embeddings.service import embedding_service
from items.models import Item, ItemEmbedding

logger = logging.getLogger(__name__)


async def engine_search(query: str, top_k: float) -> list[dict[str, ...]]:
    use_fts = settings.FULLTEXT_SEARCH_ENABLED
    use_embedding = settings.VECTOR_EMBEDDIG_ENABLED
    base_qs = ItemEmbedding.objects
    query = query.strip().lower()

    try:
        # Step 1: Apply Full-Text Search (FTS) filter if enabled
        if use_fts:
            # vectors = (
            #     SearchVector("title", "description", config=lang.strip()) for lang in settings.FULLTEXT_SEARCH_LANGUAGES
            # )
            # Dynamically combine SearchQuery objects for each language using the OR operator.
            search_queries = (
                SearchQuery(query, config=lang.strip(), search_type="websearch")
                for lang in settings.FULLTEXT_SEARCH_LANGUAGES
            )
            combined_search_query = reduce(or_, search_queries)
            # Asynchronously fetch the IDs of the top FTS candidates. This is more efficient.
            fts_candidate_ids = [
                item_id
                async for item_id in Item.objects.filter(search_vector=combined_search_query).values_list(
                    "id", flat=True
                )[: top_k * 2]
            ]
            base_qs = base_qs.filter(item_id__in=fts_candidate_ids)

            # Optimization: If FTS is the first stage and it returns no candidates, we can stop here.
            if not fts_candidate_ids:
                return []

        # Step 2: Handle the case where embedding search is disabled
        if not use_embedding:
            # If FTS was used, get the items from the ItemEmbedding query. Otherwise, do a simple icontains search.
            if use_fts:
                # We use .select_related('item') to fetch the related item efficiently
                results_qs = base_qs.select_related("item")[:top_k]
                items_iterator = (result.item async for result in results_qs)
            else:
                # Use afilter for async compatibility and slice after filtering
                items_iterator = Item.objects.afilter(
                    Q(title__icontains=query) | Q(description__icontains=query)
                ).order_by("id")[:top_k]

            items: list[dict[str, ...]] = [
                {
                    **item.__dict__,  # Unpack the item's fields
                    "distance": 0,
                }
                async for item in items_iterator
            ]
            return items

        # Step 3: Perform vector search if embedding is enabled
        if not embedding_service:
            logger.error("Embedding service not initialized")
            return []

        query_vector = await embedding_service.aget_or_create_query_embedding(query)
        if query_vector is None:
            return []

        model_name = embedding_service.backend.model_name
        results_qs = (
            base_qs.annotate(distance=CosineDistance("vector", query_vector))
            .filter(model=model_name)
            .order_by("distance")
            .select_related("item")[:top_k]
        )

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


async def engine_hybrid_search(query: str, top_k: float, alpha: float = 0.5) -> list[dict[str, ...]]:
    query = query.strip().lower()
    try:
        if not embedding_service:
            logger.error("Embedding service not initialized")
            return []

        query_vector = await embedding_service.aget_or_create_query_embedding(query)
        if query_vector is None:
            return []

        model_name = embedding_service.backend.model_name

        search_queries = (
            SearchQuery(query, config=lang.strip(), search_type="websearch")
            for lang in settings.FULLTEXT_SEARCH_LANGUAGES
        )
        combined_search_query = reduce(or_, search_queries)

        results_qs = (
            Item.objects.annotate(
                # Correctly rank the pre-calculated `search_vector` against the `SearchQuery`
                fts_rank=SearchRank("search_vector", combined_search_query, cover_density=True),
                # We use CosineDistance, which is 0 for identical, 1 for opposite.
                # So, `1 - distance` gives a similarity score from 0 to 2.
                vec_similarity=1 - CosineDistance("embedding__vector", query_vector),
            )
            .filter(Q(search_vector=combined_search_query) & Q(embedding__model=model_name))
            # Use a second annotate with F() objects instead of the legacy .extra()
            # Now we combine two similarity scores, where bigger is always better.
            .annotate(hybrid_score=(F("fts_rank") * alpha) + (F("vec_similarity") * (1 - alpha)))
            .order_by("-hybrid_score")[:top_k]
        )

        items: list[dict[str, ...]] = [
            {
                "id": item.id,
                "title": item.title,
                "description": item.description,
                "created_at": item.created_at,
                "hybrid_score": round(item.hybrid_score, 4),
            }
            async for item in results_qs
        ]
        return items

    except Exception as e:
        logger.error(f"Error in hybrid_engine_search: {e}")
        raise
