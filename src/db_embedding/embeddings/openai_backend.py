import openai
from django.conf import settings

from .backend import EmbeddingBackend


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str = None, api_key: str = None, api_base: str = None):
        self.model = model or settings.EMBEDDIG_MODEL_NAME
        if api_key:
            openai.api_key = api_key or settings.OPENAI_API_KEY
        if api_base:
            openai.api_base = api_base or settings.OPENAI_API_BASE

    def embed_text(self, text: str) -> list[float]:
        response = openai.Embedding.create(model=self.model, input=text)
        return response["data"][0]["embedding"]
