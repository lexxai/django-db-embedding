from django.conf import settings
from django.db import models
from pgvector.django import VectorField
from django.contrib.postgres.search import SearchVectorField


class Item(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Optional: full-text search vector
    search_vector = SearchVectorField(null=True, blank=True)


class ItemEmbedding(models.Model):
    item = models.OneToOneField(Item, on_delete=models.CASCADE, related_name="embedding")
    vector = VectorField(
        dimensions=settings.VECTOR_EMBEDDIG_DIMENSIONS, null=True, blank=True
    )  # OpenAI embedding vector size: 1536
    model = models.CharField(max_length=50, default=settings.EMBEDDIG_MODEL_NAME)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class QueryEmbedding(models.Model):
    query_hash = models.CharField(max_length=64, unique=True)  # SHA256
    vector = VectorField(dimensions=settings.VECTOR_EMBEDDIG_DIMENSIONS, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
