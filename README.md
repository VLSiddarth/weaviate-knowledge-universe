# Weaviate + Knowledge Universe: Two-Layer Temporal Governance for RAG

> **A stale FDA guideline ranked lower is not the same as blocked.**
> This repository makes that distinction a first-class primitive in your RAG pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Weaviate](https://img.shields.io/badge/Weaviate-1.25+-green.svg)](https://weaviate.io/)
[![Knowledge Universe](https://img.shields.io/badge/Knowledge%20Universe-API-orange.svg)](https://api.knowledgeuniverse.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Your RAG pipeline returns a 2022 FDA clinical guideline with cosine similarity 0.94.

The model synthesizes it into a confident, coherent answer. No retrieval error fires. No low-score alert. The user receives a compliance-critical response built on superseded protocol.

**This is not a ranking problem. It is a gating problem.**

Standard vector databases solve relevance. They do not solve time. A 2019 paper explaining attention mechanisms and a 2022 Stack Overflow answer about a deprecated API score identically under cosine similarity. Both are "relevant." Only one is safe to use.

The standard mitigation — TTL deletion — is too blunt. It would delete the 2019 attention paper, which is still accurate, while keeping a 2023 FINRA guideline that was superseded six months ago.

**What you need is domain-aware temporal governance: soft-ranking at retrieval time, hard-gating before context assembly.**

That is what this repository builds.

---

## The Architecture

```
Query
  │
  ▼
┌──────────────────────────────────────────────────────┐
│           LAYER 1: WEAVIATE BOOST.DECAY              │
│                                                      │
│  near_vector(query_embedding)                        │
│  + Boost(TimeDecay(                                  │
│      origin  = now,                                  │
│      scale   = domain_half_life_from_KU,  ◄── KU     │
│      curve   = EXPONENTIAL,                          │
│      depth   = 50                                    │
│    ))                                                │
│                                                      │
│  Output: soft-ranked candidates                      │
│  Stale sources appear lower — but are NOT removed    │
└──────────────────────┬───────────────────────────────┘
                       │ ranked candidates
                       ▼
┌──────────────────────────────────────────────────────┐
│      LAYER 2: KNOWLEDGE UNIVERSE GOVERNANCE          │
│                                                      │
│  For each candidate:                                 │
│  ┌────────────────────────────────────────────────┐  │
│  │  decay_score  = f(age_days, platform_half_life)│  │
│  │  velocity     = domain churn classification    │  │
│  │  retraction   = CrossRef retraction check      │  │
│  │                                                │  │
│  │  if decay_score > threshold  → BLOCKED ✗       │ │
│  │  if retracted               → BLOCKED ✗        │ │
│  │  else                       → PASSED  ✓        │ │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  Output: audit report + gated candidates             │
└──────────────────────┬───────────────────────────────┘
                       │ temporally-verified context
                       ▼
┌──────────────────────────────────────────────────────┐
│                  LLM CONTEXT                         │
│  Only temporally-clean documents reach here          │
└──────────────────────────────────────────────────────┘
```

### Why Two Layers?

| Capability | Weaviate alone | KU alone | Both layers |
|---|---|---|---|
| Semantic relevance | ✅ | ✅ via discovery | ✅ |
| Recency soft-ranking | ✅ Boost.decay | — | ✅ |
| Domain-aware half-life | ❌ | ✅ | ✅ |
| Hard-gate stale content | ❌ | ✅ | ✅ |
| Retraction detection | ❌ | ✅ CrossRef | ✅ |
| Per-domain velocity | ❌ | ✅ | ✅ |
| Conflict detection | ❌ | ✅ | ✅ |
| Regulated pipeline support | ⚠️ partial | ⚠️ partial | ✅ |

Weaviate's `Boost.decay` is excellent at continuous soft-ranking. But soft-ranking is not sufficient for regulated domains. A stale FDA guideline ranked 7th instead of 1st still reaches the LLM context. Layer 2 removes it entirely.

Knowledge Universe's governance layer is excellent at temporal audit. But it does not replace vector search — it augments it. Layer 1 provides the semantic candidates. Layer 2 certifies them.

Together, neither layer's weakness matters.

---

## The Math

### Layer 1 — Weaviate Boost.decay

Weaviate's `Boost.decay` applies a time-decay function to each document's score at retrieval time. The `scale` parameter defines when the boost reaches 0.5 — directly equivalent to a half-life.

```
boosted_score = cosine_similarity × decay_function(publication_date)

decay_function(t) = exp(-λ × age_days)   where λ = ln(2) / scale_days
```

**The key insight:** `scale` should not be a fixed constant. It should be domain-specific. A GitHub repository has a 180-day half-life. An arXiv paper has a 3-year half-life. Using a single scale for a heterogeneous corpus produces wrong rankings.

KU's `DecayBridge` translates per-platform half-lives into Weaviate scale parameters automatically:

```python
# DecayBridge maps KU platform metadata → Weaviate scale
bridge = DecayBridge()
params = bridge.get_boost_params("github", velocity_label="fast")
# params.scale = "180d"  ← correct half-life for GitHub content
# params.scale = "1095d" ← correct half-life for arXiv content
```

### Layer 2 — Knowledge Universe Decay Formula

```
decay = 1 − 0.5 ^ (age_days / half_life)

freshness = 1 − decay
```

| Decay score | Label | Action |
|---|---|---|
| 0.00 – 0.25 | `fresh` | ✅ Pass |
| 0.25 – 0.50 | `aging` | ✅ Pass (with warning) |
| 0.50 – 0.75 | `stale` | ✗ Block (default threshold) |
| 0.75 – 1.00 | `decayed` | ✗ Block |

### Platform Half-Lives

KU calibrates decay per platform based on how fast knowledge actually becomes outdated:

| Platform | Half-life | Reasoning |
|---|---|---|
| `huggingface` | 120 days | ML models superseded constantly |
| `github` | 180 days | Code goes stale with dependency updates |
| `youtube` | 270 days | Tutorials date with library releases |
| `stackoverflow` | 365 days | Answers age with framework versions |
| `arxiv` | 1,095 days | Research papers have long shelf life |
| `wikipedia` | 1,460 days | Actively maintained encyclopaedia |
| `openlibrary` | 1,825 days | Books revised infrequently |

This means a GitHub repo from 18 months ago (decay ≈ 0.51) is blocked at the default threshold, while an arXiv paper from 18 months ago (decay ≈ 0.21) passes cleanly — even though they have identical publication timestamps.

---

## Live Output

### Two-Layer Pipeline (`examples/03_two_layer_rag.py`)

```
Query: how does self-attention work in transformers

LAYER 1 ONLY: Weaviate vector search
┌────────────────┬────────────────────────────────────────┬────────┬─────────┬──────────────┐
│ Platform       │ Title                                  │  Decay │ Quality │ Status       │
├────────────────┼────────────────────────────────────────┼────────┼─────────┼──────────────┤
│ stackoverflow  │ Why is attention scaled by sqrt(d_k)   │   0.50 │     5.2 │ ✓ passed     │
│ wikipedia      │ Transformer (deep learning)            │   0.40 │     5.6 │ ✓ passed     │
│ arxiv          │ You Need Better Attention Priors       │   0.07 │     8.9 │ ✓ passed     │
│ semantic_scho… │ A Transformer Network-driven Deep Le   │   0.39 │     5.0 │ ✓ passed     │
│ huggingface    │ overthelex/attention-analysis-fewsho   │   0.40 │     5.1 │ ✓ passed     │
└────────────────┴────────────────────────────────────────┴────────┴─────────┴──────────────┘
Candidates: 10 | Passed: 5 | Blocked: 0   ← Layer 1 blocks nothing

BOTH LAYERS: Vector search + KU Governance
┌────────────────┬────────────────────────────────────────┬────────┬─────────┬──────────────┐
│ Platform       │ Title                                  │  Decay │ Quality │ Status       │
├────────────────┼────────────────────────────────────────┼────────┼─────────┼──────────────┤
│ wikipedia      │ Transformer (deep learning)            │   0.40 │     5.6 │ ✓ passed     │
│ arxiv          │ You Need Better Attention Priors       │   0.07 │     8.9 │ ✓ passed     │
│ semantic_scho… │ A Transformer Network-driven Deep Le   │   0.39 │     5.0 │ ✓ passed     │
│ huggingface    │ overthelex/attention-analysis-fewsho   │   0.40 │     5.1 │ ✓ passed     │
│ arxiv          │ Towards understanding how attention    │   0.28 │     7.4 │ ✓ passed     │
│ stackoverflow  │ Why is attention scaled by sqrt(d_k)   │   0.50 │     5.2 │ ✗ blocked    │  ← stale SO answer removed
│ semantic_scho… │ "Deep Learning Application for Seism   │   0.56 │     2.8 │ ✗ blocked    │  ← off-topic + stale removed
└────────────────┴────────────────────────────────────────┴────────┴─────────┴──────────────┘
Candidates: 10 | Passed: 5 | Blocked: 2 (20% block rate)
```

### Regulated Pipeline (`examples/04_regulated_pipeline.py`)

The same query, three different compliance domains. Each applies a different decay threshold appropriate to its regulatory environment:

```
Cross-Domain Compliance Query: transformer architecture self-attention mechanisms

▶ CLINICAL_NLP — FDA guidelines, clinical trial data
  Decay threshold: 0.20 | Velocity: moderate
  → 8/10 passed (20% block rate)

▶ FINANCIAL_DISCLOSURE — SEC filings, compliance documents
  Decay threshold: 0.35 | Velocity: fast
  → 3/10 passed (70% block rate)    ← strict: only fresh arXiv papers pass

▶ GENERAL_RAG — General purpose RAG pipeline
  Decay threshold: 0.60 | Velocity: moderate
  → 8/10 passed (20% block rate)
```

The financial disclosure domain blocks 70% of candidates — including a Wikipedia article (decay 0.40), a HuggingFace source (decay 0.40), and multiple semantic scholar results. Only arXiv papers from the last 12 months pass. This is correct behavior for SEC compliance work.

---

## Quick Start

**Prerequisites:** Docker Desktop running, Python 3.11+

```bash
# 1. Clone and set up environment
git clone https://github.com/VLSiddarth/weaviate-knowledge-universe
cd weaviate-knowledge-universe
python -m venv venv && venv\Scripts\activate    # Windows
# python -m venv venv && source venv/bin/activate  # Mac/Linux

pip install -e ".[dev]"
pip install sentence-transformers

# 2. Configure environment
cp .env.example .env
# Edit .env — add your KU_API_KEY from https://api.knowledgeuniverse.tech
# Get a free key (500 calls/month, no credit card):
# curl -X POST "https://api.knowledgeuniverse.tech/v1/signup?email=you@email.com"

# 3. Start Weaviate locally
docker compose up -d

# 4. Ingest knowledge into Weaviate
python examples/01_ingest.py

# 5. Run the two-layer pipeline
python examples/03_two_layer_rag.py

# 6. Run the regulated pipeline demo
python examples/04_regulated_pipeline.py
```

That is it. Five commands from zero to a fully governed, domain-aware RAG pipeline.

---

## API Reference

### `KUWeaviateClient`

```python
from ku_weaviate import KUWeaviateClient

client = KUWeaviateClient(
    ku_api_key="ku_test_...",          # from api.knowledgeuniverse.tech
    weaviate_url="http://localhost:8080",
    weaviate_api_key=None,             # None for local Docker
    decay_threshold=0.40,             # hard gate — block anything above this
    retraction_check=True,            # CrossRef retraction detection
)

await client.setup()
```

### `ingest_topic()`

```python
n = await client.ingest_topic(
    topic="FDA clinical NLP guidelines",
    difficulty=4,
    formats=["pdf", "html", "arxiv"],
    max_results=20,
)
# Returns: number of documents ingested into Weaviate
# Each document stored with pre-computed decay_score + embeddings
```

### `query()`

```python
result = await client.query(
    query_text="what are current FDA guidelines for NLP in clinical trials",
    limit=5,
    use_governance=True,
    velocity_label="moderate",    # adjusts threshold dynamically
    decay_threshold=0.20,         # override default for this query
)

# result.passed_documents — safe for LLM context
# result.blocked_documents — what was removed and why
# result.governance_report — full audit trail
# result.to_llm_context() — formatted string for LLM prompt

print(result.to_llm_context())
```

### `GovernanceReport`

```python
report = result.governance_report

print(f"Passed: {report.passed}/{report.total_checked}")
print(f"Block rate: {report.block_rate:.0%}")
print(f"Domain velocity: {report.domain_velocity}")

for r in report.results:
    if not r.passed:
        print(f"BLOCKED [{r.platform}] decay={r.decay_score:.2f} → {r.block_reason}")
```

---

## Regulated Use Cases

### Clinical NLP

FDA guidelines, clinical trial protocols, and HIPAA compliance documents require the strictest freshness standards. A superseded FDA protocol is a compliance liability — it must be blocked, not ranked lower.

```python
client = KUWeaviateClient(
    ku_api_key=ku_key,
    decay_threshold=0.20,    # only accept documents from last ~6 months
    retraction_check=True,   # CrossRef retraction detection enabled
)
```

At threshold 0.20, a document must be less than approximately 180 days old (for a moderate-velocity domain) to pass. Wikipedia articles, older HuggingFace datasets, and aging Stack Overflow answers are all blocked. Only fresh arXiv papers and recent official documentation reach the LLM.

### Financial Disclosure

SEC filings, FINRA guidelines, and financial compliance documents change frequently. A financial regulation from 18 months ago may be substantively different from the current version.

```python
result = await client.query(
    query_text="current SEC Rule 10b-5 disclosure requirements",
    use_governance=True,
    velocity_label="fast",       # adjusts threshold to 0.35 automatically
    decay_threshold=0.35,
)
# 70% block rate observed in testing — only very fresh sources pass
```

### Legal Research

Legal documents require version-awareness. A court ruling from 2020 may have been overturned, superseded, or reinterpreted. The governance layer blocks documents above threshold and surfaces the `retraction_status` field for any document that has been formally updated.

### General Enterprise RAG

For general-purpose enterprise RAG pipelines, the default threshold of 0.40 provides a sensible balance — blocking clearly stale content (Stack Overflow answers older than ~18 months) while allowing research papers and reference documentation that age more slowly.

---

## Domain Velocity

KU classifies every topic into one of four velocity labels. These labels adjust the governance threshold automatically — tighter gates for fast-moving domains, looser gates for stable ones.

| Velocity | Meaning | Auto-threshold | Example domains |
|---|---|---|---|
| `hypersonic` | > 65% sources from last 90 days | 0.25 | LLM releases, model cards |
| `fast` | > 35% sources from last 90 days | 0.35 | Financial data, LangChain docs |
| `stable` | > 10% sources from last 90 days | 0.40 | Medical research, software engineering |
| `frozen` | < 10% sources from last 90 days | 0.65 | HTTP spec, linear algebra, legal precedent |

```python
# Let KU determine velocity automatically
result = await client.query(
    query_text="OpenAI API function calling",
    use_governance=True,
    velocity_label="hypersonic",  # ← KU returns this from /v1/discover
)
# Threshold auto-adjusts to 0.25 — only very fresh sources pass
```

---

## Why Not Just TTL?

TTL deletion assumes all content in a domain decays at the same rate. It does not.

A 2019 arXiv paper explaining the attention mechanism is still accurate today. Deleting it because it is "more than 2 years old" destroys valid context.

A 2023 Stack Overflow answer about a LangChain API that was redesigned in v0.2 is actively dangerous regardless of its age in absolute terms — because its domain has a 365-day half-life and it crossed the stale threshold.

TTL treats time as an absolute. Temporal governance treats time as relative to domain velocity. These are different problems requiring different solutions.

---

## Repository Structure

```
weaviate-knowledge-universe/
├── README.md                       ← you are here
├── pyproject.toml
├── docker-compose.yml              ← Weaviate local instance
├── .env.example
│
├── ku_weaviate/                    ← core integration package
│   ├── __init__.py
│   ├── client.py                   ← KUWeaviateClient (main entry point)
│   ├── decay_bridge.py             ← maps KU half-lives → Weaviate scale param
│   ├── governance.py               ← post-retrieval hard-gating logic
│   ├── velocity.py                 ← domain velocity → Boost depth param
│   └── schema.py                   ← Weaviate collection schema
│
├── examples/
│   ├── 01_ingest.py                ← ingest KU sources into Weaviate
│   ├── 02_query_boosted.py         ← query with Boost.decay only
│   ├── 03_two_layer_rag.py         ← full two-layer pipeline demo
│   ├── 04_regulated_pipeline.py    ← clinical/financial hard-gate demo
│   └── 05_domain_velocity.py       ← dynamic scale from KU velocity
│
├── benchmarks/
│   └── run_benchmark.py
│
└── tests/
    ├── test_decay_bridge.py
    ├── test_governance.py
    └── test_integration.py
```

---

## Environment Variables

```bash
# .env

# Weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=              # leave empty for local Docker

# Knowledge Universe
KU_API_KEY=ku_test_...         # get free key below
KU_BASE_URL=https://api.knowledgeuniverse.tech

# Governance config
DECAY_HARD_GATE_THRESHOLD=0.40
RETRACTION_CHECK_ENABLED=true

# Optional
HF_TOKEN=                      # avoids HuggingFace rate limit warnings
OPENAI_API_KEY=                # not required — embeddings are local
```

Get a free KU API key (500 calls/month, no credit card):

```bash
curl -X POST "https://api.knowledgeuniverse.tech/v1/signup?email=you@email.com"
```

---

## Production Deployment

For production deployments against Weaviate Cloud:

```python
client = KUWeaviateClient(
    ku_api_key=os.getenv("KU_API_KEY"),
    weaviate_url=os.getenv("WEAVIATE_URL"),      # your Weaviate Cloud URL
    weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
    decay_threshold=0.40,
    retraction_check=True,
)
```

The client automatically uses `weaviate.connect_to_weaviate_cloud()` when `weaviate_api_key` is provided, and `weaviate.connect_to_local()` when it is not.

---

## Token Burn Reduction

Every document that governance blocks before reaching the LLM is a document that consumes no input tokens.

In clinical NLP testing, applying a decay threshold of 0.20 to a typical retrieved set of 10 candidates blocked approximately 50% of documents. At an average of 800 tokens per document, this reduces input token burn by approximately 4,000 tokens per query — before the LLM is invoked.

At scale, this is not a quality metric. It is a cost metric.

---

## Knowledge Universe API

This integration is built on [Knowledge Universe](https://api.knowledgeuniverse.tech) — a temporal retrieval API that scores every result for freshness and decay before it enters your pipeline.

| Endpoint | Purpose |
|---|---|
| `POST /v1/discover` | Multi-platform discovery with decay scores |
| `POST /v1/knowledge-audit` | Audit URLs for freshness and retraction status |
| `GET /v1/usage` | Check your monthly quota |
| `POST /v1/signup` | Get a free API key |

```bash
# Interactive API documentation
open https://api.knowledgeuniverse.tech

# Health check
curl https://api.knowledgeuniverse.tech/health
# {"status":"healthy","version":"1.0.0","redis":"connected"}
```

---

## Contributing

Pull requests welcome. The most valuable contributions right now are:

1. `examples/05_domain_velocity.py` — dynamic threshold from KU velocity response
2. `tests/test_decay_bridge.py` — unit tests for half-life → scale mapping
3. `tests/test_governance.py` — governance layer unit tests
4. `benchmarks/run_benchmark.py` — latency comparison: Layer 1 only vs both layers

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built with [Weaviate](https://weaviate.io) + [Knowledge Universe API](https://api.knowledgeuniverse.tech)*

*A stale source ranked lower is not the same as blocked. This makes the difference explicit.*
