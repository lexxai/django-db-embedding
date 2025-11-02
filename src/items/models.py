from django.conf import settings
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from pgvector.django import VectorField

vector_dimensions = settings.VECTOR_EMBEDDING_DIMENSIONS


class Item(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Optional: full-text search vector
    search_vector = SearchVectorField(null=True, blank=True)

    def __str__(self):
        return f"Item: {self.title}"


class ItemEmbedding(models.Model):
    item = models.OneToOneField(Item, on_delete=models.CASCADE, related_name="embedding")
    vector = VectorField(dimensions=vector_dimensions, null=True, blank=True)  # OpenAI embedding vector size: 1536
    model = models.CharField(max_length=50, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Embedding for {self.item.title}"


class QueryEmbedding(models.Model):
    query_hash = models.CharField(max_length=64, unique=True)  # SHA256
    vector = VectorField(dimensions=vector_dimensions, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Query Embedding: {self.query_hash}"
