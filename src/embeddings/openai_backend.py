import logging
from asyncio import sleep

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
        self.free_tier: bool = settings.OPENAI_API_FREE_TIER
        self.free_tier_delay: float = 60 / (settings.OPENAI_API_DELAY_TIME_RPM or 1)

    def embed_text(self, text: str) -> list[float]:
        response = async_to_sync(self.aembed_text)(text)
        return response

    async def delay_rpm(self):
        if self.free_tier:
            logger.debug(
                f"Sleep for free_tier delay: {self.free_tier_delay:.2} sec. ({settings.OPENAI_API_DELAY_TIME_RPM} RPM)"
            )
            print(
                f"delay_rpm Sleep for free_tier delay: {self.free_tier_delay:.2} sec. ({settings.OPENAI_API_DELAY_TIME_RPM} RPM)"
            )
            await sleep(self.free_tier_delay)

    async def aembed_text(self, text: str) -> list[float]:
        logger.debug(f"aembed_text text: {text}")
        await self.delay_rpm()
        try:
            response = await self.client.embeddings.create(model=self.model, input=text)
            if not response or not getattr(response, "data", None):
                logger.error(f"Invalid response: '{response}'")
                return None

            result = response.data[0].embedding
            if not result or len(result) != settings.VECTOR_EMBEDDIG_DIMENSIONS:
                logger.error(f"Invalid vector length")
            # logger.debug(f"aembed_text result: {result}")
            return result
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None
