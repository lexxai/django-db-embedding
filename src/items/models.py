from django.conf import settings
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from pgvector.django import VectorField

vector_dimensions: int = settings.VECTOR_EMBEDDING_DIMENSIONS
vector_db_field_fixed_dimensions: bool = settings.VECTOR_DB_FIELD_FIXED_DIMENSIONS


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
    vector = (
        VectorField(null=True, blank=True, dimensions=vector_dimensions)
        if vector_db_field_fixed_dimensions
        else VectorField(null=True, blank=True)
    )
    model = models.CharField(max_length=50, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Embedding for {self.item.title}"


class QueryEmbedding(models.Model):
    query_hash = models.CharField(max_length=64, unique=True)  # SHA256
    vector = (
        VectorField(null=True, blank=True, dimensions=vector_dimensions)
        if vector_db_field_fixed_dimensions
        else VectorField(null=True, blank=True)
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Query Embedding: {self.query_hash}"
