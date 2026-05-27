"""
KUWeaviateClient — Two-layer temporal governance for RAG pipelines.

Layer 1: Weaviate near_vector search (embeddings from sentence-transformers)
Layer 2: KU GovernanceLayer — post-retrieval hard-gating

No OpenAI dependency. Embeddings generated locally with all-MiniLM-L6-v2.
"""

from __future__ import annotations
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

import httpx
import weaviate
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey

from .decay_bridge import DecayBridge, WeaviateBoostParams
from .governance import GovernanceLayer, GovernanceReport
from .schema import COLLECTION_NAME, create_ku_collection

logger = logging.getLogger(__name__)

KU_BASE_URL = "https://api.knowledgeuniverse.tech"

# Lazy-loaded embedding model — loads once, reused forever
_embed_model = None


def _get_embed_model():
    """Load sentence-transformers model once. ~90MB, free, local."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model (all-MiniLM-L6-v2)...")
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✓ Embedding model loaded")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
    return _embed_model


def _embed(text: str) -> list[float]:
    """Embed a single text string. Returns 384-dim vector."""
    model = _get_embed_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


class KUWeaviateClient:
    """
    Two-layer temporal governance for RAG pipelines.

    Layer 1: Weaviate near_vector search — finds semantically similar docs
    Layer 2: KU GovernanceLayer — hard-blocks stale/retracted docs

    Usage:
        client = KUWeaviateClient(
            ku_api_key="ku_test_...",
            weaviate_url="https://xxx.weaviate.network",
            weaviate_api_key="your-weaviate-key",
        )
        await client.setup()
        await client.ingest_topic("transformer architecture", difficulty=3)
        result = await client.query("how does self-attention work")
        print(result.to_llm_context())
        await client.aclose()
    """

    def __init__(
        self,
        ku_api_key: str,
        weaviate_url: str = "http://localhost:8080",
        weaviate_api_key: Optional[str] = None,
        decay_threshold: float = 0.40,
        retraction_check: bool = True,
    ):
        self.ku_api_key = ku_api_key
        self.weaviate_url = weaviate_url
        self.decay_threshold = decay_threshold

        self._bridge = DecayBridge()
        self._governor = GovernanceLayer(
            api_key=ku_api_key,
            decay_threshold=decay_threshold,
            retraction_check=retraction_check,
        )

        # Connect to Weaviate
        if weaviate_api_key:
            logger.info(f"Connecting to Weaviate Cloud: {weaviate_url}")
            self._weaviate = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=AuthApiKey(weaviate_api_key),
            )
        else:
            logger.info("Connecting to local Weaviate Docker")
            self._weaviate = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051,
            )

        self._ku_http = httpx.AsyncClient(
            base_url=KU_BASE_URL,
            headers={
                "X-API-Key": ku_api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def setup(self):
        """Initialize Weaviate schema and embedding model."""
        create_ku_collection(self._weaviate)
        _get_embed_model()  # Pre-warm model on startup
        logger.info("KUWeaviateClient ready")

    async def ingest_topic(
        self,
        topic: str,
        difficulty: int = 3,
        formats: Optional[list[str]] = None,
        max_results: int = 20,
    ) -> int:
        """
        Discover KU sources for a topic and ingest into Weaviate
        with pre-computed embeddings.

        Returns number of documents successfully ingested.
        """
        formats = formats or ["pdf", "github", "html", "stackoverflow", "arxiv"]

        logger.info(f"Ingesting topic: '{topic}' (difficulty={difficulty})")

        resp = await self._ku_http.post(
            "/v1/discover",
            json={
                "topic": topic,
                "difficulty": difficulty,
                "formats": formats,
                "max_results": max_results,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        sources = data.get("sources", [])
        decay_scores = data.get("decay_scores", {})
        velocity = data.get("knowledge_velocity", {})
        velocity_label = (
            velocity.get("velocity_label", "unknown") if velocity else "unknown"
        )

        collection = self._weaviate.collections.get(COLLECTION_NAME)
        ingested = 0
        now = datetime.now(timezone.utc)

        with collection.batch.dynamic() as batch:
            for source in sources:
                try:
                    source_id = source.get("id", "")
                    decay_info = decay_scores.get(source_id, {})
                    decay_score = decay_info.get("decay_score", 0.4)

                    # Parse publication date — always timezone-aware
                    pub_date = source.get("publication_date")
                    pub_datetime = now
                    if pub_date:
                        try:
                            parsed = datetime.fromisoformat(
                                pub_date.replace("Z", "+00:00")
                            )
                            if parsed.tzinfo is None:
                                parsed = parsed.replace(tzinfo=timezone.utc)
                            pub_datetime = parsed
                        except ValueError:
                            pub_datetime = now

                    # Generate embedding locally (title + summary)
                    embed_text = (
                        f"{source.get('title', '')} "
                        f"{source.get('summary', '')[:500]}"
                    ).strip()
                    vector = _embed(embed_text)

                    obj = {
                        "title": source.get("title", ""),
                        "summary": source.get("summary", "")[:2000],
                        "url": source.get("url", ""),
                        "platform": source.get("source_platform", "unknown"),
                        "publication_date": pub_datetime,
                        "decay_score": float(decay_score),
                        "ku_source_id": source_id,
                        "difficulty": int(source.get("difficulty", 3)),
                        "quality_score": float(source.get("quality_score", 5.0)),
                        "open_access": bool(source.get("open_access", True)),
                        "authors": source.get("authors", []) or [],
                        "tags": source.get("tags", []) or [],
                        "ingested_at": now,
                    }

                    # Insert with our own vector
                    batch.add_object(properties=obj, vector=vector)
                    ingested += 1

                except Exception as e:
                    logger.warning(
                        f"Skipped source {source.get('id', '?')}: {e}"
                    )

        logger.info(
            f"Ingested {ingested} sources for '{topic}' "
            f"(velocity={velocity_label})"
        )
        return ingested

    async def query(
        self,
        query_text: str,
        limit: int = 10,
        platform_hint: Optional[str] = None,
        velocity_label: Optional[str] = None,
        decay_threshold: Optional[float] = None,
        use_governance: bool = True,
    ) -> "QueryResult":
        """
        Two-layer temporal query.

        Layer 1: Weaviate near_vector search using local embeddings
        Layer 2: KU GovernanceLayer hard-gating (optional)

        Returns QueryResult with passed_documents safe for LLM context.
        """
        # Get boost params from decay bridge (for metadata/logging)
        boost_params = self._bridge.get_boost_params(
            platform=platform_hint or "html",
            velocity_label=velocity_label or "unknown",
        )

        # Embed the query locally
        query_vector = _embed(query_text)

        # Layer 1: Weaviate near_vector search
        collection = self._weaviate.collections.get(COLLECTION_NAME)

        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit * 2,  # overfetch — governance will filter
            return_metadata=wvc.query.MetadataQuery(
                distance=True,
            ),
            return_properties=[
                "title", "summary", "url", "platform",
                "publication_date", "decay_score", "ku_source_id",
                "quality_score", "difficulty", "open_access",
            ],
        )

        weaviate_hits = results.objects
        urls = [
            obj.properties.get("url", "")
            for obj in weaviate_hits
            if obj.properties.get("url")
        ]

        logger.info(
            f"Weaviate: {len(weaviate_hits)} candidates "
            f"(boost scale={boost_params.scale})"
        )

        # Layer 2: KU governance using stored decay scores
        governance_report: Optional[GovernanceReport] = None

        # Build doc list from Weaviate results (decay already stored)
        all_docs = []
        for obj in weaviate_hits:
            all_docs.append({
                "url": obj.properties.get("url", ""),
                "platform": obj.properties.get("platform", "unknown"),
                "decay_score": float(
                    obj.properties.get("decay_score", 0.4) or 0.4
                ),
            })

        if use_governance and all_docs:
            governance_report = await self._governor.audit_from_weaviate_docs(
                docs=all_docs,
                domain_velocity=velocity_label,
            )
            passed_urls = set(governance_report.passed_urls)
        else:
            passed_urls = set(obj.properties.get("url", "") for obj in weaviate_hits)

        # Split into passed / blocked
        passed_docs = []
        blocked_docs = []

        for obj in weaviate_hits:
            url = obj.properties.get("url", "")
            doc = {
                "title": obj.properties.get("title", ""),
                "url": url,
                "summary": obj.properties.get("summary", ""),
                "platform": obj.properties.get("platform", ""),
                "decay_score": float(
                    obj.properties.get("decay_score", 0.4) or 0.4
                ),
                "quality_score": float(
                    obj.properties.get("quality_score", 5.0) or 5.0
                ),
                "weaviate_distance": (
                    obj.metadata.distance if obj.metadata else None
                ),
            }
            if url in passed_urls:
                passed_docs.append(doc)
            else:
                blocked_docs.append(doc)

        return QueryResult(
            query=query_text,
            total_candidates=len(weaviate_hits),
            passed_documents=passed_docs[:limit],
            blocked_documents=blocked_docs,
            boost_params=boost_params,
            governance_report=governance_report,
        )

    def close(self):
        """Sync close — for use outside async context."""
        self._weaviate.close()

    async def aclose(self):
        """Async close — always use this inside async functions."""
        self._weaviate.close()
        await self._ku_http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()


@dataclass
class QueryResult:
    """Result from KUWeaviateClient.query()"""
    query: str
    total_candidates: int
    passed_documents: list[dict]
    blocked_documents: list[dict]
    boost_params: WeaviateBoostParams
    governance_report: Optional[GovernanceReport]

    @property
    def passed_count(self) -> int:
        return len(self.passed_documents)

    @property
    def blocked_count(self) -> int:
        return len(self.blocked_documents)

    @property
    def block_rate(self) -> float:
        total = self.total_candidates
        return round(self.blocked_count / total, 3) if total else 0.0

    def to_llm_context(self, max_chars: int = 8000) -> str:
        """Format passed documents as LLM-ready context string."""
        parts = []
        total = 0
        for doc in self.passed_documents:
            chunk = (
                f"[{doc.get('platform', '').upper()}] "
                f"{doc.get('title', '')}\n"
                f"URL: {doc.get('url', '')}\n"
                f"Decay: {doc.get('decay_score', 0):.2f} | "
                f"Quality: {doc.get('quality_score', 0):.1f}\n"
                f"{(doc.get('summary', '') or '')[:500]}\n"
            )
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n---\n".join(parts)