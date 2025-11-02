import logging

from django.conf import settings
from django.core.cache import cache
from django.utils.module_loading import import_string

from items.models import QueryEmbedding
from items.utils import hash_query, ahash_query
from .backend import EmbeddingBackend

logger = logging.getLogger(__file__)


class EmbeddingService:
    def __init__(self, backend: EmbeddingBackend):
        self._backend = backend
        assert backend, "Backend must be initialized"
        self.cache_time = settings.EMBEDDING_SERVICE_CACHE_TIME

    @property
    def backend(self):
        return self._backend

    def get_or_create_query_embedding(
        self, query_text: str, input_type: EmbeddingBackend.InputType = None
    ) -> list[float] | None:
        query_text_hash = self.backend.model_name + (input_type or "") + query_text
        query_h = hash_query(query_text_hash)
        if (vector := self.cache_get(query_h)) is not None:
            return vector
        obj, created = QueryEmbedding.objects.get_or_create(query_hash=query_h)
        if created:
            obj.vector = self.backend.embed_text(query_text)
            if obj.vector is not None and len(obj.vector) == self.backend.dimensions:
                obj.save(update_fields=["vector"])
                obj.refresh_from_db(fields=["vector"])
            else:
                logger.error(f"Invalid vector length: '{obj.vector}'")
                obj.delete()
                return None
        self.cache_set(query_h, obj.vector)
        return obj.vector

    async def aget_or_create_query_embedding(
        self, query_text: str, input_type: EmbeddingBackend.InputType = None
    ) -> list[float] | None:
        query_text_hash = self.backend.model_name + (input_type or "") + query_text
        query_h = await ahash_query(query_text_hash)
        if (vector := await self.acache_get(query_h)) is not None:
            return vector
        obj, created = await QueryEmbedding.objects.aget_or_create(query_hash=query_h)
        if created:
            obj.vector = await self.backend.aembed_text(query_text, input_type)
            if obj.vector is not None and len(obj.vector) == self.backend.dimensions:
                await obj.asave(update_fields=["vector"])
                await obj.arefresh_from_db(fields=["vector"])
            else:
                logger.error(f"Invalid vector length: '{obj.vector}'")
                await obj.adelete()
                return None
        await self.acache_set(query_h, obj.vector)
        return obj.vector

    @staticmethod
    def generate_key(key):
        return "embedding_service:" + key

    async def acache_set(self, key, value):
        if self.cache_time is None or not key:
            return
        await cache.aset(self.generate_key(key), value, self.cache_time)

    async def acache_get(self, key) -> list[float] | None:
        if self.cache_time is None or not key:
            return None
        return await cache.aget(self.generate_key(key))

    def cache_set(self, key, value):
        if self.cache_time is None or not key:
            return
        cache.set(self.generate_key(key), value, self.cache_time)

    def cache_get(self, key) -> list[float] | None:
        if self.cache_time is None or not key:
            return None
        return cache.get(self.generate_key(key))


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
