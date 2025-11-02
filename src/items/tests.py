# import os
# import sys
#
# import dotenv
# from django.conf import settings
# from django.test import TestCase
# from django.test.utils import get_runner


# class ItemsTestCase(TestCase):
#     def setUp(self):
#         dotenv.load_dotenv()
#         from embeddings.service import embedding_service
#
#         self.embedding_service = embedding_service
#
#     def test_get_or_create_query_embedding(self):
#         """
#         Tests that a query embedding can be created.
#         """
#         embedding = self.embedding_service.get_or_create_query_embedding("Test")
#         self.assertIsNotNone(embedding)


if __name__ == "__main__":
    import os

    # This allows the test to be run standalone.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()

    from embeddings.service import embedding_service

    # embedding = embedding_service.get_or_create_query_embedding("Test")
    embedding = embedding_service.backend.embed_text("Test")
    print(embedding)

    # TestRunner = get_runner(settings)
    # test_runner = TestRunner()
    # # Running tests for the 'items' app.
    # failures = test_runner.run_tests(["items"])
    # sys.exit(bool(failures))
