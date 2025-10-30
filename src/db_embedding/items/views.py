from adrf.views import APIView
from asgiref.sync import sync_to_async
from rest_framework.response import Response
from pgvector.django import CosineDistance
from .models import ItemEmbedding, Item
from .serializers import ItemSerializer
from .embeddings import get_or_create_query_embedding
from django.contrib.postgres.search import SearchVector


class SemanticSearchView(APIView):

    async def get(self, request):
        query = request.query_params.get("q", "").strip()
        top_k = int(request.query_params.get("top_k", 5))

        if not query:
            return Response({"error": "Query parameter 'q' is required"}, status=400)

        # Step 1: Generate or fetch cached query embedding
        query_vector = await sync_to_async(get_or_create_query_embedding)(query)

        # Step 2: Run async ORM query
        # Optional hybrid FTS:
        # First, define the subquery for full-text search without awaiting it.
        fts_qs = Item.objects.annotate(
            search=SearchVector("title", "description")
        ).filter(search=query)

        # Then, use the subquery in the main query to run a single DB query.
        embeddings = await ItemEmbedding.objects.filter(item__in=fts_qs).annotate(
            similarity=CosineDistance("vector", query_vector)
        ).order_by("similarity").aselect_related("item")[:top_k]

        # Step 3: Serialize results
        items = [emb.item for emb in embeddings]
        serializer = ItemSerializer(items, many=True)
        return Response(serializer.data)
