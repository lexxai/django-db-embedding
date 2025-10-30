from abc import ABC, abstractmethod


class EmbeddingBackend(ABC):

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector from text."""
        ...
