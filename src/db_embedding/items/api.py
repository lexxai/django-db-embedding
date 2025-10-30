from ninja import Router
from ninja.security import django_auth


router = Router(auth=django_auth)


@router.get("/hello")
def hello(request):
    user = request.user
    return f"Hello world : {user.username}"
