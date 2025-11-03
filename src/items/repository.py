import logging
from functools import reduce
from operator import or_

from django.conf import settings
from django.contrib.postgres.search import SearchRank, SearchQuery
from django.db.models import F, Q
from pgvector.django import CosineDistance

from embeddings.backend import EmbeddingBackend
from embeddings.service import embedding_service
from items.models import Item, ItemEmbedding

logger = logging.getLogger(__name__)


async def engine_search(query: str, top_k: float, threshold: float = 1.0) -> list[dict[str, ...]]:
    query = query.strip().lower()
    if not query or len(query) < 3:
        return []
    use_fts = settings.FULLTEXT_SEARCH_ENABLED
    use_embedding = settings.VECTOR_EMBEDDIG_ENABLED
    base_qs = ItemEmbedding.objects

    try:
        fts_candidate_ids = None
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
            # If FTS finds candidates, we use them to narrow the search space for the vector search.
            # If not, we proceed with an unfiltered `base_qs` to allow for a full semantic search.
            if fts_candidate_ids:
                base_qs = base_qs.filter(item_id__in=fts_candidate_ids)

        # Step 2: Handle the case where embedding search is disabled
        if not use_embedding:
            # If FTS was used, get the items from the ItemEmbedding query. Otherwise, do a simple icontains search.
            if use_fts:
                # We use .select_related('item') to fetch the related item efficiently
                if not fts_candidate_ids:
                    return []
                results_qs = base_qs.select_related("item")[:top_k]
                items_iterator = (result.item async for result in results_qs)
            else:
                # Use afilter for async compatibility and slice after filtering
                items_iterator = Item.objects.afilter(
                    Q(title__icontains=query.lower()) | Q(description__icontains=query.lower())
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

        query_vector = await embedding_service.aget_or_create_query_embedding(
            query, embedding_service.backend.InputType.QUERY
        )
        if query_vector is None:
            return []

        model_name = embedding_service.backend.model_name
        results_qs = (
            base_qs.annotate(distance=CosineDistance("vector", query_vector))
            .filter(model=model_name)
            .filter(distance__lte=threshold)
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


async def engine_hybrid_search(
    query: str, top_k: float, alpha: float = 0.5, threshold: float = 0.0
) -> list[dict[str, ...]]:
    query = query.strip().lower()
    if not query or len(query) < 3:
        return []
    use_fts = settings.FULLTEXT_SEARCH_ENABLED
    use_embedding = settings.VECTOR_EMBEDDIG_ENABLED
    if not use_embedding or not use_fts:
        return await engine_search(query, top_k)

    try:
        if not embedding_service:
            logger.error("Embedding service not initialized")
            return []

        query_vector = await embedding_service.aget_or_create_query_embedding(
            query, embedding_service.backend.InputType.QUERY
        )
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
            # We remove the FTS filter to allow purely semantic matches to be ranked.
            .filter(embedding__model=model_name)
            # Use a second annotate with F() objects instead of the legacy .extra()
            # Now we combine two similarity scores, where bigger is always better.
            .annotate(hybrid_score=(F("fts_rank") * alpha) + (F("vec_similarity") * (1 - alpha)))
            .filter(hybrid_score__gt=threshold)
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


async def acreate_item_embeddings_in_batch(batch_data: list[dict], input_type: EmbeddingBackend.InputType = None):
    """
    Generates and saves embeddings for a batch of item data.
    `batch_data` is a list of dictionaries, each with 'id', 'title', 'description'.
    """
    if not embedding_service:
        logger.error("Embedding backend not initialized.")
        return

    input_type = input_type or embedding_service.backend.InputType.DOCUMENT

    texts_to_embed = [f"{item['title']} {item['description']}" for item in batch_data]

    # Assuming your backend has a method to embed a list of texts
    vectors = await embedding_service.backend.aget_or_create_documents_embedding(texts_to_embed, input_type=input_type)

    if not vectors or len(vectors) != len(batch_data):
        logger.error("Mismatch between number of items and generated vectors.")
        return

    # Create ItemEmbedding objects for bulk update/creation
    embeddings_to_create = []
    for i, item_data in enumerate(batch_data):
        embeddings_to_create.append(
            ItemEmbedding(item_id=item_data["id"], vector=vectors[i], model=embedding_service.backend.model_name)
        )

    # Use bulk_create for high efficiency
    await ItemEmbedding.objects.abulk_create(
        embeddings_to_create, update_conflicts=True, unique_fields=["item_id"], update_fields=["vector", "model"]
    )


#
# async def ___acreate_item_embeddings_in_batch(batch_data: list[dict], input_type: EmbeddingBackend.InputType = None):
#     """
#     Generates and saves embeddings for a batch of item data.
#     `batch_data` is a list of dictionaries, each with 'id', 'title', 'description'.
#     """
#     if not embedding_service:
#         logger.error("Embedding backend not initialized.")
#         return
#
#     input_type = input_type or embedding_service.backend.InputType.DOCUMENT
#
#     texts_to_embed = [f"{item['title']} {item['description']}" for item in batch_data]
#
#     # Assuming your backend has a method to embed a list of texts
#     vectors = await embedding_service.backend.aembed_texts(texts_to_embed, input_type=input_type)
#
#     if not vectors or len(vectors) != len(batch_data):
#         logger.error("Mismatch between number of items and generated vectors.")
#         return
#
#     # Create ItemEmbedding objects for bulk update/creation
#     embeddings_to_create = []
#     for i, item_data in enumerate(batch_data):
#         embeddings_to_create.append(
#             ItemEmbedding(item_id=item_data["id"], vector=vectors[i], model=embedding_service.backend.model_name)
#         )
#
#     # Use bulk_create for high efficiency
#     await ItemEmbedding.objects.abulk_create(
#         embeddings_to_create, update_conflicts=True, unique_fields=["item_id"], update_fields=["vector", "model"]
#     )
