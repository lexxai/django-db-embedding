import logging
from http import HTTPStatus

from django.db import OperationalError
from ninja import NinjaAPI
from ninja.renderers import BaseRenderer

try:
    import orjson as json
except ImportError:
    import json

from ninja.parser import Parser

logger = logging.getLogger()


class ORJSONParser(Parser):
    def parse_body(self, request):
        if not request.body:
            return {}
        return json.loads(request.body)


class ORJSONRenderer(BaseRenderer):
    media_type = "application/json"

    def render(self, request, data, *, response_status):
        if not data:
            return {}
        return json.dumps(data)


api = NinjaAPI(title="DataBase EMBEDDING", parser=ORJSONParser(), renderer=ORJSONRenderer())

api.add_router("/items/", "items.api.router", tags=["items"])


@api.exception_handler(OperationalError)
def db_error_handler(request, exc):
    error_str = "Database is currently unavailable. Please try again later."
    logger.error(f"{error_str} Error: {exc}")
    return api.create_response(request, {"detail": error_str}, status=HTTPStatus.SERVICE_UNAVAILABLE)


@api.get("/liveness", url_name="liveness", tags=["service"])
async def liveness(request):
    return "OK"
