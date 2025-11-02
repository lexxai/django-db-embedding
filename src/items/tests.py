if __name__ == "__main__":
    import os
    import psutil

    # This allows the test to be run standalone.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()

    import time
    from embeddings.service import embedding_service

    def get_memory_usage():
        """Returns the memory usage of the current process in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def test_instance(name: str | list[str] = "Test", batch: bool = False):
        print(name)
        mem_before = get_memory_usage()
        print(f"Memory before: {mem_before:.2f} MB")

        start_time = time.time()
        embedding = embedding_service.backend.embed_texts(name) if batch else embedding_service.backend.embed_text(name)
        end_time = time.time()

        mem_after = get_memory_usage()
        print(f"Memory after: {mem_after:.2f} MB")
        print(f"Memory used for operation: {(mem_after - mem_before):.2f} MB")

        if isinstance(embedding, list):
            if embedding and isinstance(embedding[0], list):
                for emb in embedding:
                    print("Length:", len(emb))
                    print(emb[:4])
            elif embedding:
                print("Length:", len(embedding))
                print(embedding[:4])
        print(f"Time taken to embed: {(end_time - start_time):.4f} seconds")

    test_instance("Test init")
    test_instance("Test second")
    print("\n embedding_service.backend.close()")
    embedding_service.backend.close()
    test_instance("Test reinitialize")

    test_instance(["Test list 1", "Test list 2", "Test list 3"])
    print("\nBatch mode")
    test_instance(["Test list 4", "Test list 5", "Test list 6"], batch=True)

    print("\nSleep for 30 seconds")
    time.sleep(30)
