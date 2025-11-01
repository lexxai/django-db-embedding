import logging
from asyncio import sleep

from asgiref.sync import async_to_sync
from openai import AsyncOpenAI

from django.conf import settings

from .backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, api_base: str = None):
        self.model = model or settings.OPENAI_EMBEDDIG_MODEL_NAME
        self.dimensions = dimensions or settings.VECTOR_EMBEDDIG_DIMENSIONS
        params = {}
        if api_key:
            params["api_key"] = api_key or settings.OPENAI_API_KEY
        if api_base:
            params["api_base"] = api_base or settings.OPENAI_API_BASE
        self.client = AsyncOpenAI(**params)
        self.api_delay_time_enabled: bool = settings.API_DELAY_TIME_ENABLED
        self.api_delay_time_seconds: float = 60 / (settings.API_DELAY_TIME_RPM or 1)

    @property
    def model_name(self) -> str:
        return self.model or ""

    def embed_text(self, text: str) -> list[float]:
        response = async_to_sync(self.aembed_text)(text)
        return response

    async def delay_rpm(self):
        if self.api_delay_time_enabled:
            logger.debug(
                f"Sleep for free_tier delay: {self.api_delay_time_seconds:.2} sec. ({settings.API_DELAY_TIME_RPM} RPM)"
            )
            print(
                f"delay_rpm Sleep for free_tier delay: {self.api_delay_time_seconds:.2} sec. ({settings.API_DELAY_TIME_RPM} RPM)"
            )
            await sleep(self.api_delay_time_seconds)

    async def aembed_text(self, text: str) -> list[float] | None:
        logger.debug(f"aembed_text text: {text}")
        await self.delay_rpm()
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
