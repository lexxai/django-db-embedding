import logging
from abc import ABC, abstractmethod
from asyncio import sleep as asleep
from enum import StrEnum
from time import sleep

from django.conf import settings

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    name = "abstract"

    class InputType(StrEnum):
        DOCUMENT = "search_document"
        QUERY = "search_query"

    def __init__(self, model: str = None, dimensions: int = None):
        self._client = None
        self.model = model
        self.dimensions = dimensions or settings.VECTOR_EMBEDDING_DIMENSIONS
        self.api_delay_time_enabled: bool = settings.API_DELAY_TIME_ENABLED
        self.api_delay_time_rpm = settings.API_DELAY_TIME_RPM
        self.api_delay_time_seconds: float = 60 / (self.api_delay_time_rpm or 1)

    @abstractmethod
    def get_client(self): ...

    @property
    def client(self):
        if self._client is None:
            self._client = self.get_client()
        return self._client

    @abstractmethod
    def embed_text(self, text: str, input_type: InputType = None) -> list[float]:
        """Generate embedding vector from text."""
        ...

    @abstractmethod
    async def aembed_text(self, text: str, input_type: InputType = None) -> list[float]:
        """Generate async embedding vector from text."""
        ...

    @abstractmethod
    async def aembed_texts(self, texts: list[str], input_type: InputType = None) -> list[list[float]]:
        """Generate async embedding vectors from a list of texts."""
        ...

    @property
    def model_name(self) -> str:
        if not self.model:
            return ""
        return f"{self.name}:{self.model}" or ""

    def get_prompt_name(self, input_type: InputType | None) -> str | None:
        if input_type is None:
            return None
        return str(input_type)

    async def adelay_rpm(self):
        if self.api_delay_time_enabled:
            logger.debug(
                f"Sleep for api delay: {self.api_delay_time_seconds:.2} sec. ({settings.API_DELAY_TIME_RPM} RPM)"
            )
            await asleep(self.api_delay_time_seconds)

    def delay_rpm(self):
        if self.api_delay_time_enabled:
            logger.debug(
                f"Sleep for api delay: {self.api_delay_time_seconds:.2} sec. ({settings.API_DELAY_TIME_RPM} RPM)"
            )
            sleep(self.api_delay_time_seconds)
