import httpx
from asgiref.sync import sync_to_async
from django.contrib.postgres.search import SearchVector
from django.db import IntegrityError
from django.shortcuts import aget_object_or_404, aget_list_or_404
from ninja import Router, Query
from ninja.errors import ValidationError
from ninja.security import django_auth
from pgvector.django import CosineDistance

from embeddings.service import embedding_service
from items.models import Item, ItemEmbedding
from items.schemas import ItemSchema, SearchFilters


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


@router.get("/search", response=list[ItemSchema])
async def search(request, filters: Query[SearchFilters]):
    query = filters.q
    top_k = filters.top_k

    if not query:
        raise ValidationError([{"error": "Query parameter 'q' is required"}])

    # Step 1: Generate or fetch cached query embedding
    try:
        query_vector = await embedding_service.aget_or_create_query_embedding(query)

        # Step 2: Run async ORM query
        # Optional hybrid FTS:
        # First, define the subquery for full-text search without awaiting it.
        fts_qs = Item.objects.annotate(search=SearchVector("title", "description")).filter(search=query)

        # Then, use the subquery in the main query to run a single DB query.
        embeddings = (
            await ItemEmbedding.objects.filter(item__in=fts_qs)
            .annotate(similarity=CosineDistance("vector", query_vector))
            .order_by("similarity")
            .aselect_related("item")[:top_k]
        )

        # Step 3: Serialize results
        items = [emb.item for emb in embeddings]
        return items
    except IntegrityError as e:
        raise ValidationError([{"error": str(e).split("\n")[0]}])
