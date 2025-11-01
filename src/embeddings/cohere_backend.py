import logging
from asyncio import sleep

from asgiref.sync import async_to_sync

from cohere import AsyncClientV2
from django.conf import settings

from .backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class CohereEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, api_base: str = None):
        self.model = model or settings.COHERE_EMBEDDIG_MODEL_NAME
        self.dimensions = dimensions or settings.VECTOR_EMBEDDIG_DIMENSIONS
        params = {}
        if api_key:
            params["api_key"] = api_key or settings.COHERE_API_KEY
        self.client = AsyncClientV2(**params)
        self.api_delay_time_enabled: bool = settings.API_DELAY_TIME_ENABLED
        self.api_delay_time_seconds: float = 60 / (settings.API_DELAY_TIME_RPM or 1)

    @property
    def model_name(self) -> str:
        return self.model or ""

    def embed_text(self, text: str, input_type: str = None) -> list[float]:
        response = async_to_sync(self.aembed_text)(text, input_type)
        return response

    async def delay_rpm(self):
        if self.api_delay_time_enabled:
            logger.debug(
                f"Sleep for api delay: {self.api_delay_time_seconds:.2} sec. ({settings.API_DELAY_TIME_RPM} RPM)"
            )
            await sleep(self.api_delay_time_seconds)

    async def aembed_text(self, text: str, input_type: str = None) -> list[float] | None:
        input_type = input_type or "search_query"
        logger.debug(f"aembed_text:{input_type=}, {text[:20]=} ")
        await self.delay_rpm()
        try:
            # query_input = [{"content": [{"type": "text", "text": text}]}]
            response = await self.client.embed(
                model=self.model,
                texts=[text],
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self.dimensions,
            )
            if not response or not getattr(response, "embeddings", None):
                logger.error(f"Invalid response: '{response}'")
                return None

            result = response.embeddings.float[0]
            # logger.debug(f"aembed_text result: {result}")
            if not result or len(result) != self.dimensions:
                logger.error("Invalid vector length")
            return result
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None
