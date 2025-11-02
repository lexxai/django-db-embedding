import logging

from django.conf import settings
from django.utils.module_loading import import_string

from items.models import QueryEmbedding
from items.utils import hash_query, ahash_query
from .backend import EmbeddingBackend

logger = logging.getLogger(__file__)


class EmbeddingService:
    def __init__(self, backend: EmbeddingBackend):
        self._backend = backend
        assert backend, "Backend must be initialized"

    @property
    def backend(self):
        return self._backend

    def get_or_create_query_embedding(self, query_text: str, input_type: EmbeddingBackend.InputType = None):
        query_text_hash = self.backend.model_name + (input_type or "") + query_text
        query_h = hash_query(query_text_hash)
        obj, created = QueryEmbedding.objects.get_or_create(query_hash=query_h)
        if created:
            obj.vector = self.backend.embed_text(query_text)
            if obj.vector and len(obj.vector) == self.backend.dimensions:
                obj.save()
            else:
                logger.error(f"Invalid vector length: '{obj.vector}'")
                obj.delete()
                return None
        return obj.vector

    async def aget_or_create_query_embedding(self, query_text: str, input_type: EmbeddingBackend.InputType = None):
        query_text_hash = self.backend.model_name + (input_type or "") + query_text
        query_h = await ahash_query(query_text_hash)
        obj, created = await QueryEmbedding.objects.aget_or_create(query_hash=query_h)
        if created:
            obj.vector = await self.backend.aembed_text(query_text, input_type)
            if obj.vector and len(obj.vector) == self.backend.dimensions:
                await obj.asave()
            else:
                logger.error(f"Invalid vector length: '{obj.vector}'")
                await obj.adelete()
                return None
        return obj.vector


# --------------------------
# Module-level instance (used throughout project)
# --------------------------
# A mapping of backend names to their full import paths for lazy loading.


def get_embedding_service():
    """Factory function to initialize and return the embedding service."""
    try:
        backend_name = settings.EMBEDDING_BACKEND
        backend_path = settings.EMBEDDING_BACKEND_CLASSES.get(backend_name)

        if not backend_path:
            raise ValueError(f"Invalid or unsupported embedding backend name: '{backend_name}'")

        backend_class = import_string(backend_path)
        _backend = backend_class()
        return EmbeddingService(_backend)

    except (ImportError, KeyError, ValueError, Exception) as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        return None


embedding_service = get_embedding_service()
