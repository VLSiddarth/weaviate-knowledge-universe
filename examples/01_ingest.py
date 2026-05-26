"""Ingest KU sources for a topic into Weaviate."""

import asyncio
import os
from dotenv import load_dotenv
from ku_weaviate import KUWeaviateClient

load_dotenv()

async def main():
    client = KUWeaviateClient(
        ku_api_key=os.getenv("KU_API_KEY"),
        weaviate_url=os.getenv("WEAVIATE_URL"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
    )

    try:
        await client.setup()

        topics = [
            ("transformer architecture", 3),
            ("RAG retrieval augmented generation", 3),
            ("vector database embeddings", 3),   # replaced LangChain — KU 500
            ("attention mechanism deep learning", 4),
            ("knowledge graph retrieval", 3),
        ]

        total_ingested = 0
        for topic, difficulty in topics:
            try:
                n = await client.ingest_topic(topic, difficulty=difficulty)
                print(f"✓ '{topic}': {n} sources ingested")
                total_ingested += n
            except Exception as e:
                print(f"✗ Skipped '{topic}': {e}")

        print(f"\n{'='*50}")
        print(f"Total ingested: {total_ingested} sources into Weaviate")
        print(f"{'='*50}")

    finally:
        await client.aclose()  # uses the new async close

if __name__ == "__main__":
    asyncio.run(main())