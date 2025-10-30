# items/tasks.py
import openai
from celery import shared_task
from .models import Item, ItemEmbedding

openai.api_key = "YOUR_OPENAI_API_KEY"


@shared_task
def generate_item_embedding(item_id):
    item = Item.objects.get(id=item_id)
    text = f"{item.title} {item.description}"
    response = openai.Embedding.create(model="text-embedding-3-small", input=text)
    vector = response["data"][0]["embedding"]

    ItemEmbedding.objects.update_or_create(
        item=item, defaults={"vector": vector, "model": "text-embedding-3-small"}
    )
