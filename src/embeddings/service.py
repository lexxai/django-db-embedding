import logging
import time
from types import TracebackType
from typing import TypeVar

from django.conf import settings
from django.core.cache import cache
from django.utils.module_loading import import_string

from embeddings.embedding_backend import EmbeddingBackend
from items.models import QueryEmbedding
from items.utils import hash_query

logger = logging.getLogger(__name__)


class EmbeddingService:
    _T = TypeVar("_T")

    def __init__(self, backend: EmbeddingBackend):
        self._backend = backend
        assert backend, "Backend must be initialized"
        self.cache_time = settings.EMBEDDING_SERVICE_CACHE_TIME
        self.use_redis_cache = settings.EMBEDDING_SERVICE_REDIS_CACHE_ENABLED
        self.use_sql_cache = settings.EMBEDDING_SERVICE_SQL_CACHE_ENABLED

    @property
    def backend(self):
        return self._backend

    @property
    def is_async_prefer(self) -> bool | None:
        return None if self.backend is None else self.backend.is_async_prefer

    def gen_hash_text(self, text: str, input_type: EmbeddingBackend.InputType = None):
        query_text_hash = self.backend.model_name + (str(input_type) or "") + text
        return hash_query(query_text_hash)

    def get_or_create_query_embedding(
        self, query_text: str, input_type: EmbeddingBackend.InputType = None
    ) -> list[float] | None:
        if self.backend is None:
            logger.error("Embedding backend not initialized.")
            return None
        start_time = time.time()
        query_h = self.gen_hash_text(query_text, input_type)
        if self.use_redis_cache and ((result_vector := self.cache_get(query_h)) is not None):
            duration = time.time() - start_time
            logger.debug(f"*** Time for get embed from Redis cache: {duration:.4f} seconds")
            return result_vector
        result_vector = None
        created = None
        if self.use_sql_cache:
            try:
                obj, created = QueryEmbedding.objects.get_or_create(query_hash=query_h)
                if created:
                    obj.vector = self.backend.embed_text(query_text)
                    if obj.vector is not None and 0 < len(obj.vector) <= self.backend.dimensions:
                        obj.save(update_fields=["vector"])
                        # obj.refresh_from_db(fields=["vector"])
                        result_vector = obj.vector
                    else:
                        logger.error(f"Invalid vector length: '{obj.vector}'")
                        obj.delete()
                        return None
            except Exception as e:
                logger.error(str(e))
        else:
            result_vector = self.backend.embed_text(query_text, input_type)

        if self.use_redis_cache and result_vector is not None:
            if isinstance(result_vector, list):
                import numpy as np

                self.cache_set(query_h, np.array(result_vector))
            else:
                self.cache_set(query_h, result_vector)
        duration = time.time() - start_time
        logger.debug(f"*** Time for get embed: {duration:.4f} seconds, SQL cache was {created=}")
        return result_vector

    async def aget_or_create_query_embedding(
        self, query_text: str, input_type: EmbeddingBackend.InputType = None
    ) -> list[float] | None:
        if self.backend is None:
            logger.error("Embedding backend not initialized.")
            return None
        start_time = time.time()
        query_h = self.gen_hash_text(query_text, input_type)
        if self.use_redis_cache and ((result_vector := await self.acache_get(query_h)) is not None):
            duration = time.time() - start_time
            logger.debug(f"*** Time for get embed from Redis cache: {duration:.4f} seconds")
            return result_vector
        result_vector = None
        created = None
        if self.use_sql_cache:
            try:
                obj, created = await QueryEmbedding.objects.aget_or_create(query_hash=query_h)
                if created:
                    obj.vector = await self.backend.aembed_text(query_text, input_type)
                    if obj.vector is not None and 0 < len(obj.vector) <= self.backend.dimensions:
                        await obj.asave(update_fields=["vector"])
                        # await obj.arefresh_from_db(fields=["vector"])
                        result_vector = obj.vector
                    else:
                        logger.error(f"Invalid vector length: '{obj.vector}'")
                        await obj.adelete()
                        return None
            except Exception as e:
                logger.error(str(e))
        else:
            result_vector = await self.backend.aembed_text(query_text, input_type)
        if self.use_redis_cache and result_vector is not None:
            if isinstance(result_vector, list):
                import numpy as np

                await self.acache_set(query_h, np.array(result_vector))
            else:
                await self.acache_set(query_h, result_vector)
        duration = time.time() - start_time
        logger.debug(f"*** Time for get embed: {duration:.4f} seconds, SQL cache was {created=}")
        return result_vector

    def get_or_create_documents_embedding(
        self, texts: list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[list[float]] | None:
        """
        Generates and saves embeddings for a batch of item data.
        `batch_data` is a list of dictionaries, each with 'id', 'title', 'description'.
        """
        if self.backend is None:
            logger.error("Embedding backend not initialized.")
            return None

        input_type = input_type or self.backend.InputType.DOCUMENT

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[str] = []
        texts_to_embed_id: list[int] = []
        hashes: list[str] = []
        for i, text in enumerate(texts):
            query_h = self.gen_hash_text(text, input_type)
            hashes.append(query_h)
            if self.use_redis_cache and ((vector := self.cache_get(query_h)) is not None):
                results[i] = vector
            else:
                texts_to_embed.append(text)
                texts_to_embed_id.append(i)

        vectors = self.backend.embed_texts(texts_to_embed, input_type=input_type) if texts_to_embed else None

        if vectors is None:
            return results

        for i, vector in enumerate(vectors):
            results[texts_to_embed_id[i]] = vector

            # Create ItemEmbedding objects for bulk update/creation
        embeddings_to_create = []
        import numpy as np

        for i, hashes in enumerate(hashes):
            vector = results[i]
            if self.use_sql_cache:
                embeddings_to_create.append(QueryEmbedding(query_hash=hashes, vector=vector))
            if self.use_redis_cache:
                self.cache_set(hashes, np.array(vector) if isinstance(vector, list) else vector)

        if self.use_sql_cache:
            # Use bulk_create for high efficiency
            try:
                QueryEmbedding.objects.bulk_create(
                    embeddings_to_create, update_conflicts=True, unique_fields=["query_hash"], update_fields=["vector"]
                )
            except Exception as e:
                logger.error(str(e))

        return results

    async def aget_or_create_documents_embedding(
        self, texts: list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[list[float]] | None:
        """
        Generates and saves embeddings for a batch of item data.
        """
        if self.backend is None:
            logger.error("Embedding backend not initialized.")
            return None

        input_type = input_type or self.backend.InputType.DOCUMENT

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[str] = []
        texts_to_embed_id: list[int] = []
        hashes: list[str] = []
        for i, text in enumerate(texts):
            query_h = self.gen_hash_text(text, input_type)
            hashes.append(query_h)
            if self.use_redis_cache and ((vector := await self.acache_get(query_h)) is not None):
                results[i] = vector
            else:
                texts_to_embed.append(text)
                texts_to_embed_id.append(i)

        vectors = await self.backend.aembed_texts(texts_to_embed, input_type=input_type) if texts_to_embed else None

        if vectors is None:
            return results

        for i, vector in enumerate(vectors):
            results[texts_to_embed_id[i]] = vector

            # Create ItemEmbedding objects for bulk update/creation
        embeddings_to_create = []
        import numpy as np

        for i, hashes in enumerate(hashes):
            vector = results[i]
            if self.use_sql_cache:
                embeddings_to_create.append(QueryEmbedding(query_hash=hashes, vector=vector))
            if self.use_redis_cache:
                await self.acache_set(hashes, np.array(vector) if isinstance(vector, list) else vector)

        if self.use_sql_cache:
            # Use bulk_create for high efficiency
            try:
                await QueryEmbedding.objects.abulk_create(
                    embeddings_to_create, update_conflicts=True, unique_fields=["query_hash"], update_fields=["vector"]
                )
            except Exception as e:
                logger.error(str(e))

        return results

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

    def close(self):
        if self.backend:
            self.backend.close()

    async def aclose(self):
        if self.backend:
            await self.backend.aclose()

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self: _T) -> _T:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


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
