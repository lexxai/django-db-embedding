import gc
import logging
from asyncio import sleep
from enum import StrEnum
from pathlib import Path

from asgiref.sync import sync_to_async
from django.conf import settings

from embeddings.backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingBackend(EmbeddingBackend):
    name = "huggingface"
    is_async_prefer = False

    class InputType(StrEnum):
        DOCUMENT = "document"
        QUERY = "query"

    def __init__(self, model: str = None, dimensions: int = None, **kwargs):
        model = model or settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        super().__init__(model, dimensions)
        assert self.model, "HUGGINGFACE_EMBEDDING_MODEL_NAME must be set"
        self._client = None

    def get_client(self):
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer

        cache_dir: Path = settings.EMBEDDING_MODELS_CACHE_DIR
        cache_dir.mkdir(exist_ok=True, parents=True)
        client = SentenceTransformer(self.model, cache_folder=str(cache_dir))
        # if self.dimensions:
        #     client.max_seq_length = self.dimensions

        # The first module is the tokenizer and model. If it's None, the model failed to load.
        if not client or not client._first_module():
            logger.error(f"Failed to load SentenceTransformer model: {self.model}")
            raise ValueError(
                f"Could not load SentenceTransformer model '{self.model}'. Please check the model name and configuration."
            )

        return client

    async def aclose(self):
        self.close()
        await sleep(0)

    def close(self):
        """Releases the SentenceTransformer model from memory."""
        if self._client:
            logger.debug("Releasing HuggingFace SentenceTransformer model from memory.")
            try:
                # Check if the model is on a CUDA device before trying to empty the cache
                if any(p.is_cuda for p in self._client.parameters()):
                    torch.cuda.empty_cache()  # noqa: F821
            except Exception:
                ...
            self._client = None
            gc.collect()

    def embed_text(
        self, text: str | list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[float] | list[list[float]] | None:
        if not text:
            return None
        return self.embed_texts(text, input_type=input_type)

    def embed_texts(
        self, texts: str | list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[float] | list[list[float]] | None:
        if not texts:
            return None
        if isinstance(texts, list):
            logger.debug(f"embed_texts: texts count: {len(texts)}. {input_type=} {self.get_prompt_name(input_type)}")
        else:
            logger.debug(f"embed_texts: text: {texts[:20]}. {input_type=} {self.get_prompt_name(input_type)}")

        try:
            embeddings = self.client.encode(
                texts, convert_to_tensor=False, prompt_name=self.get_prompt_name(input_type)
            )
            if embeddings is None:
                logger.error("Invalid embeddings")
                return None

            result = embeddings.tolist()
            if not result:
                logger.error(f"Invalid result {embeddings}")
                return None

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
