import logging

from django.db.utils import OperationalError
from django.http import JsonResponse

logger = logging.getLogger()


class DatabaseTimeoutMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            response = self.get_response(request)
        except OperationalError as e:
            if "couldn't get a connection" in str(e):
                error_str = "Database is currently unavailable. Please try again later."
                logger.error(error_str)
                return JsonResponse(
                    {"error": error_str},
                    status=503,
                )
            raise
        return response
