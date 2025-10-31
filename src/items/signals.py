from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.db import transaction
from .models import Item
from .tasks import generate_item_embedding, update_item_vector_search


@receiver(post_save, sender=Item)
def item_post_save_receiver(sender, instance, created, update_fields, **kwargs):
    """
    Handles post-save signals for Items to enqueue embedding and search vector updates.
    """
    # Determine if the text content has changed, which requires re-processing.
    # If update_fields is None, it means a full .save() was called, so we assume a change.
    if update_fields is None:
        text_content_changed = True
    else:
        text_content_changed = "title" in update_fields or "description" in update_fields

    # We only proceed if the item was just created or if its text content has changed.
    if not created and not text_content_changed:
        return

    def enqueue_tasks():
        if settings.VECTOR_EMBEDDIG_ENABLED:
            generate_item_embedding.delay(instance.id)
        if settings.FULLTEXT_SEARCH_ENABLED:
            update_item_vector_search.delay(instance.id)

    # Enqueue tasks only after the database transaction has been successfully committed.
    transaction.on_commit(enqueue_tasks)
