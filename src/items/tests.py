# Create your tests here.
if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    from openai import OpenAI

    client = OpenAI()

    response = client.embeddings.create(
        model="text-embedding-3-small", input="Write a short bedtime story about a unicorn."
    )

    print(response)
