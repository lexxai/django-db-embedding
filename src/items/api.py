from asgiref.sync import sync_to_async
from django.shortcuts import aget_object_or_404, aget_list_or_404
from ninja import Router, Query
from ninja.errors import ValidationError
from ninja.security import django_auth

from items.models import Item
from items.repository import engine_search, engine_hybrid_search
from items.schemas import ItemSchema, SearchFilters, SearchResultSchema, HybridSearchFilters, HybridSearchResultSchema


async def adjango_auth(request):
    return await sync_to_async(django_auth)(request)


router = Router()


@router.get("", response=list[ItemSchema])
async def list_items(request):
    items = await aget_list_or_404(Item)
    return items


@router.get("/{id}/", response=ItemSchema)
async def get_item(request, id: int):
    items = await aget_object_or_404(Item, pk=id)
    return items


@router.get("/search", response=list[SearchResultSchema])
async def search(request, filters: Query[SearchFilters]):
    try:
        return await engine_search(**filters.dict())
    except Exception as e:
        raise ValidationError([{"error": str(e).split("\n")[0]}])


@router.get("/hybrid-search", response=list[HybridSearchResultSchema])
async def hybrid_search_unified(request, filters: Query[HybridSearchFilters]):
    return await engine_hybrid_search(**filters.dict())
