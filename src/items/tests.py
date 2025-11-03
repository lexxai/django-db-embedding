import logging

if __name__ == "__main__":
    import os
    import psutil

    # This allows the test to be run standalone.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    import time
    from embeddings.service import embedding_service

    def get_memory_usage():
        """Returns the memory usage of the current process in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def test_instance(name: str | list[str] = "Test"):
        logger.info("-" * 60)
        logger.info(f"*** Input text: '{name}'")
        logger.info(f"*** Model name: '{embedding_service.backend.model_name}'")
        mem_before = get_memory_usage()
        logger.info(f"*** Memory before: {mem_before:.2f} MB")

        start_time = time.time()
        embedding = (
            embedding_service.backend.embed_texts(name)
            if isinstance(name, list)
            else embedding_service.backend.embed_text(name)
        )
        end_time = time.time()

        mem_after = get_memory_usage()
        logger.info(f"*** Memory after: {mem_after:.2f} MB")
        logger.info(f"*** Memory used for operation: {(mem_after - mem_before):.2f} MB")

        if isinstance(embedding, list):
            if embedding and isinstance(embedding[0], list):
                for i, emb in enumerate(embedding):
                    logger.info(f"Length of vector [{i}]: {len(emb)}")
                    logger.info(f"{emb[:4]}...")
            elif embedding:
                logger.info("Length of vector: %s", len(embedding))
                logger.info(f"{embedding[:4]}...")
        logger.info(f"*** Time taken to embed: {(end_time - start_time):.4f} seconds")

    test_instance("Test init")
    test_instance("Test second")
    logger.info("******** Embedding service closing now for free resources")
    embedding_service.close()
    test_instance("Test reinitialize")
    logger.info("*** Batch mode")
    test_instance(["Test list 1", "Test list 2", "Test list 3", "Test list 4", "Test list 5", "Test list 6"])

    logger.info("*** Sleep for 30 seconds")
    time.sleep(30)
