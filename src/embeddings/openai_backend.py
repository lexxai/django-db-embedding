import logging

from asgiref.sync import async_to_sync
from openai import AsyncOpenAI

from django.conf import settings

from .backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str = None, api_key: str = None, api_base: str = None):
        self.model = model or settings.EMBEDDIG_MODEL_NAME
        params = {}
        if api_key:
            params["api_key"] = api_key or settings.OPENAI_API_KEY
        if api_base:
            params["api_base"] = api_base or settings.OPENAI_API_BASE
        self.client = AsyncOpenAI(**params)

    def embed_text(self, text: str) -> list[float]:
        response = async_to_sync(self.aembed_text)(text)
        return response

    async def aembed_text(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(model=self.model, input=text)
        result = response["data"][0]["embedding"]
        if len(result) != settings.VECTOR_EMBEDDIG_DIMENSIONS:
            logger.error(f"Invalid vector length: '{result}'")
        logger.debug(f"aembed_text result: {result}")
        return result
