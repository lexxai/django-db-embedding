# embeddings/service.py
import logging

from django.conf import settings


from items.models import QueryEmbedding
from items.utils import hash_query, ahash_query
from .backend import EmbeddingBackend


logger = logging.getLogger(__file__)


class EmbeddingService:
    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend

    def get_or_create_query_embedding(self, query_text: str):
        query_h = hash_query(query_text)
        obj, created = QueryEmbedding.objects.get_or_create(query_hash=query_h)
        if created:
            obj.vector = self.backend.embed_text(query_text)
            if obj.vector and len(obj.vector) == settings.VECTOR_EMBEDDIG_DIMENSIONS:
                obj.save()
            else:
                logger.error(f"Invalid vector length: '{obj.vector}'")
                obj.delete()
                return None
        return obj.vector

    async def aget_or_create_query_embedding(self, query_text: str, input_type: str = None):
        query_text_hash = self.backend.model_name + input_type + query_text
        query_h = await ahash_query(query_text_hash)
        obj, created = await QueryEmbedding.objects.aget_or_create(query_hash=query_h)
        if created:
            obj.vector = await self.backend.aembed_text(query_text, input_type)
            if obj.vector and len(obj.vector) == settings.VECTOR_EMBEDDIG_DIMENSIONS:
                await obj.asave()
            else:
                logger.error(f"Invalid vector length: '{obj.vector}'")
                await obj.adelete()
                return None
        return obj.vector


# --------------------------
# Module-level instance (used throughout project)
# --------------------------
try:
    backend = None
    backend_name = settings.EMBEDDING_BACKEND
    match backend_name:
        case "openai":
            from .openai_backend import OpenAIEmbeddingBackend

            backend = OpenAIEmbeddingBackend()
        case "cohere":
            from .cohere_backend import CohereEmbeddingBackend

            backend = CohereEmbeddingBackend()

    if not backend:
        raise Exception("Invalid backend name")
    embedding_service = EmbeddingService(backend)
except Exception as e:
    backend = None
    embedding_service = None
    logger.error(f"Failed to initialize embedding service: {e}")
#
