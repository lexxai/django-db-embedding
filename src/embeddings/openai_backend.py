import logging

import httpx
from asgiref.sync import async_to_sync
from django.conf import settings

from embeddings.backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"

    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, base_url: str = None, **kwargs):
        model = model or settings.OPENAI_EMBEDDIG_MODEL_NAME
        super().__init__(model, dimensions)
        assert self.model, "OPENAI_EMBEDDIG_MODEL_NAME must be set"
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    def get_client(self):
        from openai import AsyncOpenAI

        params = {}
        if self.api_key:
            params["api_key"] = self.api_key or settings.OPENAI_API_KEY
        if self.base_url:
            params["base_url"] = self.base_url or settings.OPENAI_BASE_URL

        if proxy := getattr(settings, "HTTPX_PROXY_SERVER", None):
            params["http_client"] = httpx.AsyncClient(
                base_url=params.get("base_url", ""),
                proxy=proxy,
                http2=getattr(settings, "HTTPX_HTTP2_ENABLED", True),
                timeout=self.kwargs.get("timeout"),
            )
        else:
            params["http_client"] = httpx.AsyncClient(
                base_url=params.get("base_url", ""),
                http2=getattr(settings, "HTTPX_HTTP2_ENABLED", True),
                timeout=self.kwargs.get("timeout"),
            )
        self.kwargs.pop("http_client", None)

        client = AsyncOpenAI(**params, **self.kwargs)
        assert client, "Failed to initialize OpenAI client"
        return client

    def embed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        response = async_to_sync(self.aembed_text)(text)
        return response

    def embed_texts(self, texts: list[str], input_type: EmbeddingBackend.InputType = None) -> list[list[float]] | None:
        response = async_to_sync(self.aembed_texts)(texts)
        return response

    async def aembed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        logger.debug(f"aembed_text text: {text}")
        await self.adelay_rpm()
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

    async def aembed_texts(
        self, texts: list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[list[float]] | None:
        logger.debug(f"aembed_texts texts count: {len(texts)}")
        await self.adelay_rpm()
        try:
            response = await self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)
            if not response or not getattr(response, "data", None):
                logger.error(f"Invalid response: '{response}'")
                return None

            results = [item.embedding for item in response.data]
            return results
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None
