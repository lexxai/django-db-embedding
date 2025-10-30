import datetime

# from django.contrib.postgres.search import SearchVectorField
from ninja import Schema, Field

from items.models import Item


class ItemSchema(Schema):
    id: int = Field()
    title: str = Field()
    description: str = Field()
    created_at: datetime.datetime = Field()

    # search_vector: SearchVectorField

    class Meta:
        model = Item
