import gc
import inspect
import logging
from abc import ABC, abstractmethod

from asgiref.sync import sync_to_async, async_to_sync
from django.conf import settings

logger = logging.getLogger(__name__)


class BaseClient(ABC):
    """Abstract base class for clients."""

    name = "abstract"
    is_async_prefer = True

    def __init__(self, preload: bool = False, **kwargs):
        self._client = None
        self.api_key = kwargs.pop("api_key", None)
        self.base_url = kwargs.pop("base_url", None)
        self.api_delay_time_enabled: bool = settings.API_DELAY_TIME_ENABLED
        self.api_delay_time_rpm = settings.API_DELAY_TIME_RPM
        self.api_delay_time_seconds: float = 60 / (self.api_delay_time_rpm or 1)
        self.kwargs = kwargs
        if preload:
            self.get_client()

    @abstractmethod
    def get_client(self): ...

    @property
    def client(self):
        if self._client is None:
            self._client = self.get_client()
        return self._client

    def close(self):
        if self._client:
            close_method = getattr(self._client, "close", getattr(self._client, "aclose", None))
            if close_method:
                if inspect.iscoroutinefunction(close_method):
                    try:
                        async_to_sync(close_method)()
                    except Exception:
                        ...
                else:
                    # Run synchronous close in a thread to avoid blocking the event loop
                    close_method()
            self._client = None
            gc.collect()

    async def aclose(self):
        """Asynchronously close the client"""
        if self._client:
            aclose_method = getattr(self._client, "aclose", None)
            if aclose_method:
                if inspect.iscoroutinefunction(aclose_method):
                    await aclose_method()
                    self._client = None
                    gc.collect()
                    return None
            close_method = getattr(self._client, "close", None)
            if close_method:
                if inspect.iscoroutinefunction(aclose_method):
                    await aclose_method()
                else:
                    await sync_to_async(close_method)()
                self._client = None
                gc.collect()
                return None
        return None
