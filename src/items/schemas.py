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


class SearchFilters(Schema):
    query: str = Field(..., min_length=1, max_length=255, alias="q")
    top_k: int = Field(5, ge=1, le=10, description="Number of results to return")
    threshold: float = Field(1.0, description="Threshold for cosine distance.")


class HybridSearchFilters(SearchFilters):
    alpha: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Alpha value for hybrid search. "
        "If value is 1.0 only then semantic of vector search, if 0.0 then only FTS search.",
    )
    threshold: float = Field(0.0, description="Threshold for hybrid score.")


class SearchResultSchema(ItemSchema):
    distance: float = Field(0, description="Distance, 0 close, 1 far")


class HybridSearchResultSchema(ItemSchema):
    hybrid_score: float = Field(0, description="Hybrid score")
