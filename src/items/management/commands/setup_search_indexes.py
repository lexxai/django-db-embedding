from django.core.management.base import BaseCommand
from django.db import connection
from django.conf import settings
from items.models import Item, ItemEmbedding

# Get table and column names dynamically from the models
VECTOR_TABLE_NAME = ItemEmbedding._meta.db_table
VECTOR_COLUMN = "vector"
TSVECTOR_TABLE_NAME = Item._meta.db_table
TSVECTOR_COLUMN = "search_vector"
DIMENSION = settings.VECTOR_EMBEDDIG_DIMENSIONS


class Command(BaseCommand):
    help = "Create FTS trigger, GIN index, and vector index for hybrid search"

    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            self.stdout.write("🔧 Setting up hybrid search infrastructure...")

            # 1. Create FTS trigger function (multilingual: English + Ukrainian)
            cursor.execute(
                f"""
                CREATE OR REPLACE FUNCTION {TSVECTOR_TABLE_NAME}_search_vector_trigger()
                RETURNS trigger AS $$
                BEGIN
                    NEW.{TSVECTOR_COLUMN} :=
                        setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B') ||
                        setweight(to_tsvector('ukrainian', coalesce(NEW.title, '')), 'A') ||
                        setweight(to_tsvector('ukrainian', coalesce(NEW.description, '')), 'B');
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql;
            """
            )
            self.stdout.write(self.style.SUCCESS("FTS trigger function created"))

            # 2. Attach trigger to table
            cursor.execute(
                f"""
                DROP TRIGGER IF EXISTS {TSVECTOR_TABLE_NAME}_search_vector_update ON {TSVECTOR_TABLE_NAME};
                CREATE TRIGGER {TSVECTOR_TABLE_NAME}_search_vector_update
                BEFORE INSERT OR UPDATE OF title, description
                ON {TSVECTOR_TABLE_NAME}
                FOR EACH ROW EXECUTE FUNCTION {TSVECTOR_TABLE_NAME}_search_vector_trigger();
            """
            )
            self.stdout.write(self.style.SUCCESS("FTS trigger attached"))

            # 3. Create GIN index on search_vector
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {TSVECTOR_TABLE_NAME}_{TSVECTOR_COLUMN}_idx
                ON {TSVECTOR_TABLE_NAME} USING GIN ({TSVECTOR_COLUMN});
            """
            )
            self.stdout.write(self.style.SUCCESS("GIN index created"))

            # 4. Create vector index (IVFFlat)
            # Estimate lists: ~rows / 1000, but min 10, max 1000
            cursor.execute(f"SELECT COUNT(*) FROM {VECTOR_TABLE_NAME};")
            row_count = cursor.fetchone()[0] or 1
            lists = min(max(10, row_count // 1000), 1000)

            # Optional: Use HNSW if supported (pgvector >= 0.5 + PG >= 16)
            use_hnsw = getattr(settings, "USE_HNSW_INDEX", False)

            if use_hnsw:
                # HNSW is more accurate & faster, but requires newer pgvector
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {VECTOR_TABLE_NAME}_{VECTOR_COLUMN}_hnsw_idx
                    ON {VECTOR_TABLE_NAME} USING hnsw ({VECTOR_COLUMN} vector_cosine_ops);
                """
                )
                self.stdout.write(self.style.SUCCESS("HNSW vector index created (lists auto-tuned: N/A)"))
            else:
                # IVFFlat: good balance of speed & recall
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {VECTOR_TABLE_NAME}_{VECTOR_COLUMN}_ivfflat_idx
                    ON {VECTOR_TABLE_NAME} USING ivfflat ({VECTOR_COLUMN} vector_cosine_ops)
                    WITH (lists = {lists});
                """
                )
                self.stdout.write(self.style.SUCCESS(f"IVFFlat vector index created (lists = {lists})"))

            # 5. Optional: Update existing rows (if search_vector is NULL)
            cursor.execute(
                f"""
                UPDATE {TSVECTOR_TABLE_NAME}
                SET {TSVECTOR_COLUMN} = 
                    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(description, '')), 'B') ||
                    setweight(to_tsvector('ukrainian', coalesce(title, '')), 'A') ||
                    setweight(to_tsvector('ukrainian', coalesce(description, '')), 'B')
                WHERE {TSVECTOR_COLUMN} IS NULL;
            """
            )
            self.stdout.write(self.style.SUCCESS("Existing rows updated (if needed)"))

        self.stdout.write(self.style.SUCCESS("🎉 Hybrid search infrastructure ready!"))
