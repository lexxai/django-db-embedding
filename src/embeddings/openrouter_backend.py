import logging
from enum import StrEnum

from asgiref.sync import async_to_sync
from django.conf import settings

from clients.httpx_client import HttpxClient
from embeddings.embedding_backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class OpenRouterEmbeddingBackend(EmbeddingBackend):
    name = "openrouter"
    is_async_prefer = True
    clientClass = HttpxClient

    class InputType(StrEnum):
        DOCUMENT = "document"
        QUERY = "query"
        RETRIEVAL_DOCUMENT = "query"
        RETRIEVAL_QUERY = "document"

    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, **kwargs):
        model = model or settings.OPENROUTER_EMBEDDIG_MODEL_NAME
        self.base_url = settings.OPENROUTER_BASE_URL
        self.kwargs = kwargs
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.kwargs["api_key"] = self.api_key
        self.kwargs["base_url"] = self.base_url
        super().__init__(model, dimensions, **kwargs)
        assert self.model, "OPENROUTER_EMBEDDIG_MODEL_NAME must be set"

        self._client = None

    def embed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        response = async_to_sync(self.aembed_text)(text, input_type)
        return response

    def embed_texts(self, texts: list[str], input_type: EmbeddingBackend.InputType = None) -> list[list[float]] | None:
        response = async_to_sync(self.aembed_texts)(texts)
        return response

    async def aembed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        input_type = input_type or self.InputType.QUERY
        logger.debug(f"aembed_text:{input_type=}, {text[:20]=} ")
        embeddings = await self.aembed_texts([text], input_type=input_type)
        return embeddings[0] if embeddings is not None else None

    async def aembed_texts(
        self, texts: list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[list[float]] | None:
        input_type = input_type or self.InputType.DOCUMENT
        logger.debug(f"aembed_texts:{input_type=}, texts count: {len(texts)}")
        await self.client.adelay_rpm()
        try:
            async with self.client.aclient() as client:
                json = {
                    "model": self.model,
                    "input": texts,
                    "input_type": str(input_type),
                    "dimensions": self.dimensions,
                    "encoding_format": "float",
                }
                logger.debug(f"aembed_texts: json: {json}")
                response = await client.post("/embeddings", json=json)
            if response.status_code != 200:
                raise ValueError(f"Invalid response: '{response.text[:100]}'")

            response_data = response.json()
            if not response_data or "data" not in response_data:
                raise ValueError(f"Invalid response not data: '{response.text[:100]}'")
            if not (embeddings := response_data.get("data")):
                raise ValueError(f"Invalid response data is empty: '{response.text[:100]}'")

            embedding = [e for item in embeddings if (e := item.get("embedding"))]
            return embedding
        except ValueError as e:
            logger.error(f"[{self.model}] {e}")
            return None
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None
