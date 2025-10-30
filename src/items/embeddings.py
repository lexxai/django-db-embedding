import openai
from django.conf import settings

from .models import QueryEmbedding
from .utils import ahash_query


async def get_or_create_query_embedding(query_text: str):
    query_h = await ahash_query(query_text)
    obj, created = await QueryEmbedding.objects.aget_or_create(query_hash=query_h)
    if created:
        response = await openai.Embedding.acreate(model=settings.EMBEDDIG_MODEL_NAME, input=query_text)
        obj.vector = response["data"][0]["embedding"]
        await obj.asave()
    return obj.vector
