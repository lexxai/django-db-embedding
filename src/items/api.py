import httpx
from asgiref.sync import sync_to_async
from django.contrib.postgres.search import SearchVector
from django.db import IntegrityError
from django.shortcuts import aget_object_or_404, aget_list_or_404
from ninja import Router, Query, Schema
from ninja.errors import ValidationError
from ninja.security import django_auth
from pgvector.django import CosineDistance

from embeddings.service import embedding_service
from items.models import Item, ItemEmbedding
from items.schemas import ItemSchema, SearchFilters, SearchResultSchema


async def django_aauth(request):
    return await sync_to_async(django_auth)(request)


router = Router(auth=django_aauth)


@router.get("/hello")
async def hello(request):
    user = request.user
    return f"Hello world : {user.username}"


@router.get("", response=list[ItemSchema])
async def list_items(request):
    items = await aget_list_or_404(Item)  # supported if ORM async ready
    return items


@router.get("/{id}/", response=ItemSchema)
async def get_item(request, id: int):
    items = await aget_object_or_404(Item, pk=id)
    return items


@router.get("/external")
async def call_external(request):
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.github.com")
    return {"status": r.status_code}


@router.get("/search", response=list[SearchResultSchema])
async def search(request, filters: Query[SearchFilters]):
    query = filters.q
    top_k = filters.top_k
    use_fts = False

    if not query:
        raise ValidationError([{"error": "Query parameter 'q' is required"}])

    # Step 1: Generate or fetch cached query embedding
    try:
        query_vector = await embedding_service.aget_or_create_query_embedding(query)

        # Step 2: Run async ORM query
        # Optional hybrid FTS:
        # First, define the subquery for full-text search without awaiting it.
        if use_fts:
            fts_qs = Item.objects.annotate(search=SearchVector("title", "description")).filter(search=query)
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
        items = [
            {
                **emb.item.__dict__,  # Unpack the item's fields
                "distance": emb.distance,
            }
            async for emb in results_qs
        ]
        return items
    except Exception as e:
        raise ValidationError([{"error": str(e).split("\n")[0]}])
