import logging

from django.conf import settings

from clients.base_client import BaseClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseClient):
    name = "openai"

    def __init__(self, is_async: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.is_async = is_async

    def get_client(self):
        if self.is_async:
            from openai import AsyncOpenAI as OpenAIClientClass
            from httpx import AsyncClient as HttpxClient
        else:
            from openai import OpenAI as OpenAIClientClass
            from httpx import Client as HttpxClient

        params = {}
        if self.api_key:
            params["api_key"] = self.api_key or settings.OPENAI_API_KEY
        if self.base_url:
            params["base_url"] = self.base_url or settings.OPENAI_BASE_URL

        if proxy := getattr(settings, "HTTPX_PROXY_SERVER", None):
            params["http_client"] = HttpxClient(
                base_url=params.get("base_url", ""),
                proxy=proxy,
                http2=getattr(settings, "HTTPX_HTTP2_ENABLED", True),
                timeout=self.kwargs.get("timeout"),
            )
        else:
            params["http_client"] = HttpxClient(
                base_url=params.get("base_url", ""),
                http2=getattr(settings, "HTTPX_HTTP2_ENABLED", True),
                timeout=self.kwargs.get("timeout"),
            )
        self.kwargs.pop("http_client", None)

        client = OpenAIClientClass(**self.kwargs, **params)
        assert client, f"Failed to initialize OpenAI {'Async' if self.is_async else ''} client"
        return client
