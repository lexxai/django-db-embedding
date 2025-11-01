from django.contrib import admin

from items.models import ItemEmbedding, QueryEmbedding, Item


@admin.register(Item)
class ItemAdmin(admin.ModelAdmin):
    """Admin configuration for the Item model."""

    list_display = ("id", "title", "created_at", "updated_at")
    search_fields = ("title", "description")
    list_filter = ("created_at",)
    date_hierarchy = "created_at"
    readonly_fields = (
        "search_vector",
        "created_at",
        "updated_at",
    )


@admin.register(ItemEmbedding)
class ItemEmbeddingAdmin(admin.ModelAdmin):
    """Admin configuration for the ItemEmbedding model."""

    list_display = ("id", "item_title", "model", "created_at")
    search_fields = ("item__title",)
    list_filter = ("model", "created_at")
    raw_id_fields = ("item",)
    # We replace 'vector' with our custom 'vector_display' method
    readonly_fields = (
        "item_title",
        "vector_display",
        "created_at",
        "updated_at",
    )
    exclude = ("vector", "item")

    @admin.display(description="Item Title", ordering="item__title")
    def item_title(self, obj):
        return obj.item.title

    @admin.display(description="Vector Summary")
    def vector_display(self, obj):
        """Creates a more readable summary for the vector field."""
        if hasattr(obj, "vector") and obj.vector is not None and len(obj.vector) > 0:
            return f"Dimensions: {len(obj.vector)} | Preview: [{', '.join(map(str, obj.vector[:3]))}...]"
        return None


@admin.register(QueryEmbedding)
class QueryEmbeddingAdmin(admin.ModelAdmin):
    """Admin configuration for the QueryEmbedding model."""

    list_display = ("id", "query_hash", "created_at")
    search_fields = ("query_hash",)
    readonly_fields = (
        "query_hash",
        "vector_display",
        "created_at",
    )
    fields = ("query_hash", "vector_display", "created_at")

    @admin.display(description="Vector Summary")
    def vector_display(self, obj):
        """Creates a more readable summary for the vector field."""
        if hasattr(obj, "vector") and obj.vector is not None and len(obj.vector) > 0:
            return f"Dimensions: {len(obj.vector)} | Preview: [{', '.join(map(str, obj.vector[:3]))}...]"
        return None
