import logging

from asgiref.sync import async_to_sync
from django.conf import settings

from clients.base_client import BaseClient
from clients.openai_client import OpenAIClient
from embeddings.embedding_backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"
    clientClass = OpenAIClient

    def __init__(
        self, model: str = None, dimensions: int = None, preload: bool = False, client: BaseClient = None, **kwargs
    ):
        model = model or settings.OPENAI_EMBEDDIG_MODEL_NAME
        super().__init__(model, dimensions, preload, client, **kwargs)
        assert self.model, "OPENAI_EMBEDDIG_MODEL_NAME must be set"

    def get_client(self):
        if self._client is None:
            self._client = self.clientClass(is_async=self.is_async_prefer, **self.kwargs)
            assert self._client, "Failed to initialize {clientClass.name} client"
        return self._client

    def embed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        response = async_to_sync(self.aembed_text)(text)
        return response

    def embed_texts(self, texts: list[str], input_type: EmbeddingBackend.InputType = None) -> list[list[float]] | None:
        response = async_to_sync(self.aembed_texts)(texts)
        return response

    async def aembed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        logger.debug(f"aembed_text text: {text}")
        await self.client.adelay_rpm()
        try:
            response = await self.client.embeddings.create(model=self.model, input=text, dimensions=self.dimensions)
            if not response or not getattr(response, "data", None):
                logger.error(f"Invalid response: '{response}'")
                return None

            result = response.data[0].embedding
            if not result or len(result) != self.dimensions:
                logger.error("Invalid vector length")
            # logger.debug(f"aembed_text result: {result}")
            return result
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None

    async def aembed_texts(
        self, texts: list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[list[float]] | None:
        logger.debug(f"aembed_texts texts count: {len(texts)}")
        await self.client.adelay_rpm()
        try:
            response = await self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)
            if not response or not getattr(response, "data", None):
                logger.error(f"Invalid response: '{response}'")
                return None

            results = [item.embedding for item in response.data]
            return results
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None
