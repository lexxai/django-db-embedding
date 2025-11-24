import logging

import httpx
from django.conf import settings

from clients.base_client import BaseClient

logger = logging.getLogger(__name__)


class HttpxClient(BaseClient):
    name = "httpx"
    clientClass: type["httpx.AsyncClient"] | type["httpx.Client"] = None

    def __init__(self, is_async: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.is_async = is_async
        try:
            if self.is_async:
                from httpx import AsyncClient as Client

                self.clientClass: type["httpx.AsyncClient"]
            else:
                from httpx import Client

                self.clientClass: type["httpx.Client"]
        except ImportError:
            raise ImportError("httpx is not installed. Please install it with `uv add httpx`.")
        self.clientClass = Client

    def get_client(self) -> "httpx.AsyncClient | httpx.Client":
        params = {"headers": {"Content-Type": "application/json"}}
        if self.api_key:
            params["headers"]["Authorization"] = f"Bearer {self.api_key}"
        if self.base_url:
            params["base_url"] = self.base_url
        params["http2"] = getattr(settings, "HTTPX_HTTP2_ENABLED", True)
        params["timeout"] = self.kwargs.get("timeout")
        if proxy := getattr(settings, "HTTPX_PROXY_SERVER", None):
            params["proxy"] = proxy

        client = self.clientClass(**params)
        assert client, "Failed to initialize HTTPX client"
        return client
