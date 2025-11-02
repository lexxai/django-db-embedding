import logging

from asgiref.sync import async_to_sync
from django.conf import settings

from embeddings.backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"

    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, api_base: str = None):
        model = model or settings.OPENAI_EMBEDDIG_MODEL_NAME
        super().__init__(model, dimensions)
        assert self.model, "OPENAI_EMBEDDIG_MODEL_NAME must be set"
        self.api_key = api_key
        self.api_base = api_base

    def get_client(self):
        from openai import AsyncOpenAI

        params = {}
        if self.api_key:
            params["api_key"] = self.api_key or settings.COHERE_API_KEY
        if self.api_base:
            params["base_url"] = self.api_base or settings.COHERE_API_BASE
        client = AsyncOpenAI(**params)
        assert client, "Failed to initialize OpenAI client"
        return client

    def embed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        response = async_to_sync(self.aembed_text)(text)
        return response

    def embed_texts(self, texts: list[str], input_type: EmbeddingBackend.InputType = None) -> list[list[float]] | None:
        response = async_to_sync(self.aembed_texts)(texts)
        return response

    async def aembed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        logger.debug(f"aembed_text text: {text}")
        await self.adelay_rpm()
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
        await self.adelay_rpm()
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
