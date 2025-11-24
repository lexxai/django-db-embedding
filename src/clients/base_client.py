import gc
import inspect
import logging
from abc import ABC, abstractmethod
from asyncio import sleep as asleep
from contextlib import asynccontextmanager
from time import sleep

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

    @asynccontextmanager
    async def aclient(self):
        client = self.get_client()
        try:
            yield client
        finally:
            await client.aclose()

    def __getattr__(self, name):
        """Delegate attribute access to the underlying client."""
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.client, name)

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

    async def adelay_rpm(self):
        if self.api_delay_time_enabled:
            logger.debug(
                f"Sleep for api delay: {self.api_delay_time_seconds:.2} sec. ({settings.API_DELAY_TIME_RPM} RPM)"
            )
            await asleep(self.api_delay_time_seconds)

    def delay_rpm(self):
        if self.api_delay_time_enabled:
            logger.debug(
                f"Sleep for api delay: {self.api_delay_time_seconds:.2} sec. ({settings.API_DELAY_TIME_RPM} RPM)"
            )
            sleep(self.api_delay_time_seconds)
