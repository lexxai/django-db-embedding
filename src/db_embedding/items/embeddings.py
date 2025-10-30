from .models import QueryEmbedding
from .utils import hash_query
import openai


def get_or_create_query_embedding(query_text: str):
    query_h = hash_query(query_text)
    obj, created = QueryEmbedding.objects.get_or_create(query_hash=query_h)
    if created:
        response = openai.Embedding.create(
            model="text-embedding-3-small", input=query_text
        )
        obj.vector = response["data"][0]["embedding"]
        obj.save()
    return obj.vector
