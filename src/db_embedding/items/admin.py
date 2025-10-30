from django.contrib import admin

from items.models import ItemEmbedding, QueryEmbedding, Item

admin.site.register(Item)
admin.site.register(ItemEmbedding)
admin.site.register(QueryEmbedding)
