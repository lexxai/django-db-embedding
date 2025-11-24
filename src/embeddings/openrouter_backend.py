import logging

from django.conf import settings

from embeddings.openai_backend import OpenAIEmbeddingBackend

logger = logging.getLogger(__name__)


class OpenRouterEmbeddingBackend(OpenAIEmbeddingBackend):
    name = "openrouter"

    def __init__(
        self,
        model: str = None,
        dimensions: int = None,
        preload: bool = False,
        api_key: str = None,
        base_url: str = None,
        **kwargs,
    ):
        model = model or settings.OPENROUTER_EMBEDDIG_MODEL_NAME
        api_key = api_key or settings.OPENROUTER_API_KEY
        base_url = base_url or settings.OPENROUTER_BASE_URL
        super().__init__(model, dimensions, preload=preload, api_key=api_key, base_url=base_url, **kwargs)
