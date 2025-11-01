from abc import ABC, abstractmethod


class EmbeddingBackend(ABC):

    @abstractmethod
    def embed_text(self, text: str, input_type: str = None) -> list[float]:
        """Generate embedding vector from text."""
        ...

    @abstractmethod
    async def aembed_text(self, text: str, input_type: str = None) -> list[float]:
        """Generate async embedding vector from text."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...
