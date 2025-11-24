import logging
from abc import ABC, abstractmethod
from enum import StrEnum

from django.conf import settings

from clients.base_client import BaseClient

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    name = "abstract"
    is_async_prefer = True
    clientClass = None

    class InputType(StrEnum):
        DOCUMENT = "search_document"
        QUERY = "search_query"
        RETRIEVAL_QUERY = "Retrieval-query"
        RETRIEVAL_DOCUMENT = "Retrieval-document"

    PROMPTS = {}

    def __init__(
        self, model: str = None, dimensions: int = None, preload: bool = False, client: BaseClient = None, **kwargs
    ):
        self._client = client
        self.model: str = model
        self.dimensions: int = dimensions or settings.VECTOR_EMBEDDING_DIMENSIONS
        self.kwargs = kwargs
        if preload or settings.EMBEDDING_BACKEND_PRELOAD:
            self.get_client()

    def get_client(self):
        if self._client is None:
            assert self.clientClass, "clientClass is not defined"
            self._client = self.clientClass(is_async=self.is_async_prefer, **self.kwargs)
            assert self._client, "Failed to initialize {clientClass.name} client"
        return self._client

    @property
    def client(self):
        if self._client is None:
            self._client = self.get_client()
        return self._client

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    def embed_text(self, text: str, input_type: InputType = None) -> list[float]:
        """Generate embedding vector from text."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str], input_type: InputType = None) -> list[list[float]]:
        """Generate embedding vectors from a list of texts."""
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

    def generate_prompt(self, input_type: InputType, style: str, title: str = None) -> str:
        prompt = self.PROMPTS.get(style, {}).get(input_type, "")
        if prompt:
            prompt = prompt.format(title=title)
        return prompt
