import httpx
from asgiref.sync import sync_to_async
from django.shortcuts import aget_object_or_404, aget_list_or_404
from ninja import Router
from ninja.security import django_auth

from items.models import Item
from items.schemas import ItemSchema


async def django_aauth(request):
    return await sync_to_async(django_auth)(request)


router = Router(auth=django_aauth)


@router.get("/hello")
def hello(request):
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
