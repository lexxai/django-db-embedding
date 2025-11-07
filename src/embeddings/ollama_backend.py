import logging

from django.conf import settings

from embeddings.openai_backend import OpenAIEmbeddingBackend

logger = logging.getLogger(__name__)


class OllamaEmbeddingBackend(OpenAIEmbeddingBackend):
    name = "ollama"

    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, base_url: str = None, **kwargs):
        model = model or settings.OLLAMA_EMBEDDIG_MODEL_NAME
        super().__init__(model, dimensions)
        assert self.model, "OLLAMA_EMBEDDIG_MODEL_NAME must be set"
        self.api_key = api_key or settings.OLLAMA_API_KEY
        assert self.api_key, "OLLAMA_API_KEY must be set"
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        assert self.base_url, "OLLAMA_API_BASE must be set"
        self.kwargs = kwargs
