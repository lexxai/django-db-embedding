from http import HTTPStatus

from django.db import OperationalError
from ninja import NinjaAPI
import logging

logger = logging.getLogger()
api = NinjaAPI()

api.add_router("/items/", "items.api.router", tags=["items"])


@api.exception_handler(OperationalError)
def db_error_handler(request, exc):
    error_str = "Database is currently unavailable. Please try again later."
    logger.error(f"{error_str} Error: {exc}")
    return api.create_response(
        request, {"detail": error_str}, status=HTTPStatus.SERVICE_UNAVAILABLE
    )


@api.get("/liveness", url_name="liveness", tags=["service"])
def liveness(request):
    return "OK"
