# AI Exercise - Retrieval

> simple RAG example

## Project requirements

### uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to install and manage python dependencies.

### Docker Engine (optional)

Install [Docker Engine](https://docs.docker.com/engine/install/) to build and run the API's Docker image locally.

## Installation

```bash
make install
```

## API

The project includes an API built with [FastAPI](https://fastapi.tiangolo.com/). Its code can be found at `src/api`.

The API is containerized using a [Docker](https://docs.docker.com/get-started/) image, built from the `Dockerfile` and `docker-compose.yml` at the root. This is optional, you can also run the API without docker.

### Environment Variables

Copy .env_example to .env and fill in the values.

### Build and start the API

To build and start the API, use the following Makefile command:

```bash
make dev-api
```

you can also use `make start-api` to start the API using Docker.

## Frontend

The project includes a frontend built with [Streamlit](https://streamlit.io/). Its code can be found at `demo`.

Run the frontend with:

```bash
make start-app
```

## Testing

To run unit tests, run `pytest` with:

```bash
make test
```

## Evaluation

The project includes a comprehensive evaluation framework for assessing retrieval quality.

### Run Evaluation

**Via API endpoint:**
```bash
# Start the API first (make dev-api)
curl http://localhost/evaluate
```

**Via CLI tool:**
```bash
# Run evaluation with default settings
uv run python -m ai_exercise.evaluation.run_evaluation

# Specify custom K neighbors
uv run python -m ai_exercise.evaluation.run_evaluation --k 10

# Save results to custom path
uv run python -m ai_exercise.evaluation.run_evaluation --output my_results.json
```

### Evaluation Metrics

The evaluation framework measures:

- **Precision@K** - Proportion of retrieved items that are relevant
- **Recall@K** - Proportion of relevant items that were retrieved
- **F1@K** - Harmonic mean of Precision and Recall
- **Mean Reciprocal Rank (MRR)** - Quality of ranking (1/rank of first relevant item)
- **NDCG@K** - Normalized Discounted Cumulative Gain (position-weighted relevance)
- **Hit Rate@K** - Whether at least one relevant item appears in top-K

Metrics are calculated at K ∈ {1, 3, 5, 10} for comprehensive analysis.

### Test Dataset

Test queries are defined in `ai_exercise/evaluation/test_queries.json` with:
- 30 diverse test queries
- Categories: endpoint discovery, schema lookup, authentication, filtering/pagination
- Ground truth relevant chunks for each query

### Understanding Results

**Good retrieval system indicators:**
- MRR > 0.8 (first relevant result appears early)
- Precision@5 > 0.6 (most top-5 results are relevant)
- Hit Rate@5 = 1.0 (all queries find at least one relevant result)
- NDCG@5 > 0.7 (good ranking quality)

**Areas needing improvement:**
- Low MRR → Relevant results ranked too low
- Low Precision → Too many irrelevant results
- Low Recall → Missing relevant documents
- Low Hit Rate → Some queries find nothing relevant

## Formatting and static analysis

There is some preset up formatting and static analysis tools to help you write clean code. check the make file for more details.

```bash
make lint
```

```bash
make format
```

```bash
make typecheck
```

# Get Started

Have a look in `ai_exercise/constants.py`. Then check out the server routes in `ai_exercise/main.py`.

1. Load some documents by calling the `/load` endpoint. Does the system work as intended? Are there any issues?

2. Find some method of evaluating the quality of the retrieval system.

3. See how you can improve the retrieval system. Some ideas:
- Play with the chunking logic
- Try different embeddings models
- Other types of models which may be relevant
- How else could you store the data for better retrieval?

---

## Retrieval System Improvements & Production Readiness

### Phase 1: Evaluation Framework ✅ IMPLEMENTED

A comprehensive evaluation system has been built with:
- **30 test queries** covering diverse use cases (endpoint discovery, schemas, auth, etc.)
- **6 metrics** (Precision@K, Recall@K, F1@K, MRR, NDCG@K, Hit Rate@K)
- **API endpoint** (`/evaluate`) and **CLI tool** for running evaluations
- **Category-wise analysis** to identify weak areas

### Phase 2: Suggested Retrieval Quality Improvements

#### 2.1 Re-ranking
**What:** Add a second-stage re-ranker after initial retrieval
**Why:** Embedding models optimize for broad similarity; re-rankers optimize for precise relevance
**How to implement:**
- Use cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-12-v2`)
- After retrieving top-K candidates, re-score with cross-encoder
- Return top-N after re-ranking

**Expected impact:** +10-20% MRR, +15-25% Precision@5

#### 2.2 Hybrid Search (Semantic + Keyword)
**What:** Combine vector similarity with keyword matching (BM25)
**Why:** Catches queries with specific technical terms that embeddings might miss
**How to implement:**
- Index documents with BM25 (using `rank-bm25` library)
- Retrieve top-K from both vector store and BM25
- Combine scores using weighted fusion (e.g., 0.7 × semantic + 0.3 × keyword)

**Expected impact:** +5-10% Recall@5, better on technical queries

#### 2.3 Metadata Filtering & Pre-filtering
**What:** Filter by service, method type, or other metadata before similarity search
**Why:** Reduces search space, improves precision when user specifies context
**How to implement:**
- Extract service names from query ("HRIS", "ATS", "CRM")
- Extract HTTP methods from query ("GET", "POST", "DELETE")
- Use ChromaDB's `where` parameter to pre-filter metadata

**Expected impact:** +20-30% Precision when applicable

#### 2.4 Query Expansion
**What:** Expand query with synonyms and related terms
**Why:** Improves recall by catching different phrasings
**How to implement:**
- Add domain-specific synonyms (e.g., "candidate" → "applicant", "employee" → "worker")
- Embed expanded query and original query, average embeddings

**Expected impact:** +5-10% Recall@5

#### 2.5 Tuning Distance Threshold
**What:** Lower the distance threshold from 2.0 (very permissive) to ~0.8-1.2
**Why:** Current threshold may include many irrelevant results
**How to implement:**
- Run evaluation at different thresholds (0.5, 0.8, 1.0, 1.2, 1.5)
- Choose threshold that maximizes F1@5

**Expected impact:** +10-15% Precision with minimal Recall loss

#### 2.6 Better Embeddings Model
**What:** Upgrade from `text-embedding-3-small` to `text-embedding-3-large`
**Why:** Larger model captures more nuanced semantic relationships
**How to implement:**
- Change `embeddings_model` in constants.py
- Re-embed all documents

**Expected impact:** +3-8% across all metrics (marginal but consistent improvement)

### Phase 3: Production Readiness Improvements

#### 3.1 Robustness & Reliability

**Retry Logic with Exponential Backoff**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_embedding(text):
    return openai_client.embeddings.create(...)
```
**Where to add:** Wrap all OpenAI API calls (embeddings, completions)

**Fallback LLM Providers**
```python
# Try OpenAI → Anthropic → Local model
try:
    response = openai_client.chat.completions.create(...)
except OpenAIError:
    response = anthropic_client.messages.create(...)
```
**Where to add:** `ai_exercise/llm/completions.py`

**Circuit Breaker Pattern**
- Fail fast when external service is down
- Use library like `pybreaker`
- Prevents cascading failures

**Rate Limiting**
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/chat")
@limiter.limit("10/minute")
def chat_route(...):
    ...
```

**Caching**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding_cached(text):
    return get_embedding(text)
```
**Where to add:** Cache query embeddings, LLM responses for repeated queries

**Timeouts**
```python
response = openai_client.chat.completions.create(
    ...,
    timeout=30.0  # 30 second timeout
)
```
**Where to add:** All external API calls

#### 3.2 Monitoring & Observability

**Structured Logging**
```python
import structlog

logger = structlog.get_logger()
logger.info("retrieval_completed",
            query=query,
            num_results=len(results),
            latency_ms=latency)
```

**Metrics Collection**
- Track: P50/P95/P99 latency, error rate, cache hit rate
- Tools: Prometheus + Grafana, or Datadog

**Distributed Tracing**
- Use OpenTelemetry to trace request flow
- Identify bottlenecks (embedding, retrieval, LLM calls)

**Alerting Rules**
- Alert if error rate > 5%
- Alert if P95 latency > 5s
- Alert if vector store query fails

#### 3.3 Scalability

**Async Operations**
```python
@app.post("/chat")
async def chat_route(chat_query: ChatQuery) -> ChatOutput:
    # Use asyncio.gather for parallel operations
    rewrite_task = asyncio.create_task(get_completion_async(...))
    retrieval_task = asyncio.create_task(retrieve_chunks_async(...))

    rewritten_query, chunks = await asyncio.gather(rewrite_task, retrieval_task)
```

**Batch Processing**
- Batch embed multiple documents at once (already done in `add_documents`)
- Use OpenAI batch API for cheaper embeddings

**Vector Store Optimization**
- Enable HNSW index parameters tuning in ChromaDB
- Consider switching to purpose-built vector DBs (Pinecone, Weaviate, Qdrant) for scale

**Load Balancing**
- Deploy multiple API instances behind load balancer
- Use container orchestration (Kubernetes, ECS)

#### 3.4 Data Quality

**Chunk Validation**
```python
def validate_chunk(chunk: str) -> bool:
    # Reject chunks that are too short or too long
    return 50 <= len(chunk) <= 2000
```

**Duplicate Detection**
```python
import hashlib

def chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# Track hashes to avoid duplicate chunks
```

**Freshness Checks**
```python
# Store OpenAPI spec ETags, re-fetch when changed
# Scheduled job to check for updates daily
```

**Schema Validation**
- Validate OpenAPI specs before chunking
- Use `openapi-spec-validator` library

### Implementation Priority

**High Priority (Week 1-2):**
1. ✅ Evaluation framework (DONE)
2. Retry logic + timeouts
3. Structured logging
4. Distance threshold tuning

**Medium Priority (Week 3-4):**
5. Hybrid search (BM25 + semantic)
6. Re-ranking with cross-encoder
7. Caching layer
8. Async operations

**Low Priority (Month 2+):**
9. Metadata filtering
10. Query expansion
11. Fallback LLM providers
12. Advanced monitoring/tracing

### Success Metrics

Track improvements via evaluation framework:
- **Baseline:** Run evaluation now to get current metrics
- **Target:** Achieve MRR > 0.85, Precision@5 > 0.70, Hit Rate@5 = 1.0
- **Measure:** Re-run evaluation after each improvement
- **A/B Test:** Compare old vs new approaches on same test set