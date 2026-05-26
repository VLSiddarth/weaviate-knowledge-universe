"""
Weaviate Collection Schema

No external vectorizer — embeddings generated locally via sentence-transformers
and stored as pre-computed vectors. This removes OpenAI dependency entirely
and gives us full control over the embedding model.

Model: all-MiniLM-L6-v2 (384 dimensions, ~90MB, runs locally for free)
"""

from __future__ import annotations
import weaviate
import weaviate.classes as wvc

COLLECTION_NAME = "KUSource"
VECTOR_DIMENSIONS = 384  # all-MiniLM-L6-v2


def create_ku_collection(client: weaviate.WeaviateClient) -> None:
    """
    Create the KUSource collection with no vectorizer.
    Vectors are pre-computed by us and inserted with each object.
    """
    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping.")
        return

    client.collections.create(
        name=COLLECTION_NAME,
        description="Knowledge Universe sources with temporal decay metadata",
        # No vectorizer — we supply our own vectors at insert time
        properties=[
            wvc.config.Property(
                name="title",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="summary",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="url",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="platform",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="publication_date",
                data_type=wvc.config.DataType.DATE,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="decay_score",
                data_type=wvc.config.DataType.NUMBER,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="ku_source_id",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="difficulty",
                data_type=wvc.config.DataType.INT,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="quality_score",
                data_type=wvc.config.DataType.NUMBER,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="open_access",
                data_type=wvc.config.DataType.BOOL,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="authors",
                data_type=wvc.config.DataType.TEXT_ARRAY,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="tags",
                data_type=wvc.config.DataType.TEXT_ARRAY,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="ingested_at",
                data_type=wvc.config.DataType.DATE,
                skip_vectorization=True,
            ),
        ],
    )
    print(f"✓ Created collection '{COLLECTION_NAME}'")