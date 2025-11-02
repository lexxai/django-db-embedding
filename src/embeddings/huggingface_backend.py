import logging
from enum import StrEnum

import torch  # noqa: F401
from asgiref.sync import sync_to_async
from django.conf import settings
from sentence_transformers import SentenceTransformer

from .backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingBackend(EmbeddingBackend):
    name = "huggingface"

    class InputType(StrEnum):
        DOCUMENT = "document"
        QUERY = "query"

    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, api_base: str = None):
        model = model or settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        super().__init__(model, dimensions)
        # Construct the cache directory path relative to the project root
        self.cache_dir = settings.EMBEDDING_MODELS_CACHE_DIR
        self.cache_dir.makedirs(self.cache_dir, exist_ok=True, parents=True)

        self.client = SentenceTransformer(model, cache_folder=str(self.cache_dir))
        if dimensions:
            self.client.max_seq_length = dimensions

    def embed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        logger.debug(f"embed_text: text[:20]={text[:20]}")
        try:
            embedding = self.client.encode(text, convert_to_tensor=False, prompt_name=self.get_prompt_name(input_type))
            if embedding is None:
                logger.error("Invalid embedding")
                return None
            result = embedding.tolist()
            if not result or (len(result) > self.dimensions):
                logger.error("Invalid vector length")
            return result
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None

    def embed_texts(self, texts: list[str], input_type: EmbeddingBackend.InputType = None) -> list[list[float]] | None:
        logger.debug(f"embed_texts: texts count: {len(texts)}")
        try:
            embeddings = self.client.encode(
                texts, convert_to_tensor=False, prompt_name=self.get_prompt_name(input_type)
            )
            if embeddings is None:
                logger.error("Invalid embeddings")
                return None
            result = embeddings.tolist()
            if not result or (len(result) != len(texts)):
                logger.error("Invalid result length")
                return None
            for embedding in result:
                if not embedding or len(embedding) > self.dimensions:
                    logger.error("Invalid vector length")
            return result
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None

    async def aembed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        logger.debug("async mode")
        return await sync_to_async(self.embed_text)(text, input_type=input_type)

    async def aembed_texts(
        self, texts: list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[list[float]] | None:
        logger.debug("async mode")
        return await sync_to_async(self.embed_texts)(texts, input_type=input_type)
