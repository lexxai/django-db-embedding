from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Item
from .tasks import generate_item_embedding


@receiver(post_save, sender=Item)
def enqueue_item_embedding(sender, instance, created, **kwargs):
    if created:
        generate_item_embedding.apply_async(args=[instance.id])
