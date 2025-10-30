# embeddings/service.py

from items.models import QueryEmbedding
from items.utils import hash_query, ahash_query
from .backend import EmbeddingBackend
from .openai_backend import OpenAIEmbeddingBackend


class EmbeddingService:
    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend

    def get_or_create_query_embedding(self, query_text: str):
        query_h = hash_query(query_text)
        obj, created = QueryEmbedding.objects.get_or_create(query_hash=query_h)
        if created:
            obj.vector = self.backend.embed_text(query_text)
            obj.save()
        return obj.vector

    async def aget_or_create_query_embedding(self, query_text: str):
        query_h = await ahash_query(query_text)
        obj, created = await QueryEmbedding.objects.aget_or_create(query_hash=query_h)
        if created:
            obj.vector = await self.backend.aembed_text(query_text)
            await obj.asave()
        return obj.vector


# --------------------------
# Module-level instance (used throughout project)
# --------------------------
backend = OpenAIEmbeddingBackend()
embedding_service = EmbeddingService(backend)
