from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Creates the vector extension in the database."

    def handle(self, *args, **options): ...
