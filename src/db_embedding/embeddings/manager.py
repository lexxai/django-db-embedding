# embeddings/manager.py
from django.conf import settings

from .openai_backend import OpenAIEmbeddingBackend
from .service import EmbeddingService


class EmbeddingManager:
    _service = None

    @classmethod
    def get_service(cls):
        if cls._service is None:
            backend = OpenAIEmbeddingBackend(api_key=settings.OPENAI_API_KEY)
            cls._service = EmbeddingService(backend)
        return cls._service
