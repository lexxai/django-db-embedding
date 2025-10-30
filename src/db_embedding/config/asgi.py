"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Get the default Django ASGI application.
# This handles all the HTTP requests.
http_app = get_asgi_application()


async def application(scope, receive, send):
    """
    The main ASGI application.
    This wrapper adds lifespan protocol support to the default Django ASGI application.
    """
    if scope["type"] == "http":
        # Let the default Django app handle HTTP requests.
        await http_app(scope, receive, send)

    elif scope["type"] == "lifespan":
        # Handle lifespan events (startup and shutdown).
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                # You can put startup logic here, e.g., initializing connection pools.
                print("INFO:     Application startup complete.")
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                # You can put shutdown logic here, e.g., cleaning up resources.
                print("INFO:     Application shutdown complete.")
                await send({"type": "lifespan.shutdown.complete"})
                return
    else:
        # Optionally handle other protocols here, or raise an error.
        # For now, we'll just ignore them.
        pass
