from rest_framework.views import APIView
from rest_framework.response import Response
from pgvector.django import CosineDistance
from .models import ItemEmbedding, Item
from .serializers import ItemSerializer
from .embeddings import get_or_create_query_embedding
from django.contrib.postgres.search import SearchVector
import asyncio


class SemanticSearchView(APIView):

    async def get(self, request):
        query = request.query_params.get("q", "").strip()
        top_k = int(request.query_params.get("top_k", 5))

        if not query:
            return Response({"error": "Query parameter 'q' is required"}, status=400)

        # Step 1: Generate or fetch cached query embedding
        query_vector = await asyncio.to_thread(get_or_create_query_embedding, query)

        # Step 2: Run synchronous ORM in thread
        def fetch_results():
            # Optional hybrid FTS:
            qs = Item.objects.annotate(
                search=SearchVector("title", "description")
            ).filter(search=query)
            return (
                ItemEmbedding.objects.filter(item__in=qs)
                .annotate(similarity=CosineDistance("vector", query_vector))
                .order_by("similarity")[:top_k]
            )

        embeddings = await asyncio.to_thread(fetch_results)

        # Step 3: Serialize results
        items = [emb.item for emb in embeddings]
        serializer = ItemSerializer(items, many=True)
        return Response(serializer.data)
