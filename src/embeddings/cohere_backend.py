import logging

from asgiref.sync import async_to_sync
from cohere import AsyncClientV2
from django.conf import settings

from embeddings.backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class CohereEmbeddingBackend(EmbeddingBackend):
    name = "cohere"

    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, api_base: str = None):
        model = model or settings.COHERE_EMBEDDIG_MODEL_NAME
        super().__init__(model, dimensions)
        self.api_key = api_key
        self.api_base = api_base

    def get_client(self):
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key or settings.COHERE_API_KEY
        if self.api_base:
            params["base_url"] = self.api_base or settings.COHERE_API_BASE
        client = AsyncClientV2(**params)
        return client

    def embed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        response = async_to_sync(self.aembed_text)(text, input_type)
        return response

    def embed_texts(self, texts: list[str], input_type: EmbeddingBackend.InputType = None) -> list[list[float]] | None:
        response = async_to_sync(self.aembed_texts)(texts)
        return response

    async def aembed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        input_type = input_type or self.InputType.QUERY
        logger.debug(f"aembed_text:{input_type=}, {text[:20]=} ")
        await self.adelay_rpm()
        try:
            response = await self.client.embed(
                model=self.model,
                texts=[text],
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self.dimensions,
            )
            if not response or not getattr(response, "embeddings", None):
                logger.error(f"Invalid response: '{response}'")
                return None

            result = response.embeddings.float[0]
            if not result or len(result) != self.dimensions:
                logger.error("Invalid vector length")
            return result
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None

    async def aembed_texts(
        self, texts: list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[list[float]] | None:
        input_type = input_type or self.InputType.DOCUMENT
        logger.debug(f"aembed_texts:{input_type=}, texts count: {len(texts)}")
        await self.adelay_rpm()
        try:
            response = await self.client.embed(
                model=self.model,
                texts=texts,
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self.dimensions,
            )
            if not response or not getattr(response, "embeddings", None):
                logger.error(f"Invalid response: '{response}'")
                return None

            results = response.embeddings.float
            return results
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None
