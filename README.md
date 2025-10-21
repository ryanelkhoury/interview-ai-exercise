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

## Assignment Submission

### Implementation Summary

#### ✅ Multi-Spec Knowledge Base
Modified the RAG system to load all 7 StackOne OpenAPI specifications through the `/load` endpoint:
- `stackone.json` - Core StackOne API
- `hris.json` - Human Resources Information System
- `ats.json` - Applicant Tracking System
- `lms.json` - Learning Management System
- `iam.json` - Identity & Access Management
- `crm.json` - Customer Relationship Management
- `marketing.json` - Marketing automation
PLEASE NOTE THIS MAY TAKE ~30 MINUTES.

**Implementation:** [`ai_exercise/loading/document_loader.py:159-181`](ai_exercise/loading/document_loader.py#L159-L181)

#### ✅ Graceful Limitation Handling
The system now gracefully indicates when it cannot find relevant answers:
- Distance threshold filtering prevents returning irrelevant results
- System prompt explicitly instructs: "If you don't find the answer, say: 'I couldn't find that in the documentation.'"
- Retrieved chunks are filtered based on embedding similarity scores

**Implementation:** [`ai_exercise/main.py:178-197`](ai_exercise/main.py#L178-L197)

#### ✅ Evaluation Framework
Built a comprehensive Ragas-based evaluation system:
- **32 test queries** covering LMS, IAM, CRM, and Marketing APIs
- **4 Ragas metrics:**
  - Context Precision (are retrieved docs relevant?)
  - Context Recall (did we retrieve all necessary info?)
  - Faithfulness (is the answer faithful to context?)
  - Answer Relevancy (does the answer address the question?)
- **Ground truth validation** for each test query
- **Easy execution:** Run with `make evaluate`
- **Documentation:** See [`evaluation_guide.md`](evaluation_guide.md) for detailed metrics explanation

**Implementation:** [`ai_exercise/evaluate_ragas.py`](ai_exercise/evaluate_ragas.py)

#### ✅ Retrieval Quality Improvements (Stretch Goals)

**1. Contextual Retrieval**
- Uses LLM to generate situating context for each chunk before embedding
- Provides document-level context (API name, version, servers, schemas, paths) to improve semantic search accuracy
- Based on Anthropic's contextual retrieval technique
- **Implementation:** [`ai_exercise/loading/document_loader.py:19-62`](ai_exercise/loading/document_loader.py#L19-L62)

**2. Query Rewriting**
- Rewrites user queries to be more explicit and standalone
- Resolves pronouns and vague references from conversation history
- Improves retrieval by making queries self-contained
- **Implementation:** [`ai_exercise/main.py:126-149`](ai_exercise/main.py#L126-L149)

**3. Smart Chunking Strategy**
- Separate chunks for endpoints (one per method/path combination)
- Separate chunks for each schema definition
- Separate chunk for security schemes
- Preserves complete information per logical unit (no data loss)
- **Implementation:** [`ai_exercise/loading/document_loader.py:64-139`](ai_exercise/loading/document_loader.py#L64-L139)

**4. Distance Threshold Filtering**
- Filters retrieved chunks by embedding distance (cosine similarity)
- Prevents returning irrelevant results even if they're in top-K
- Configurable threshold via `SETTINGS.distance_threshold`
- **Implementation:** [`ai_exercise/main.py:178`](ai_exercise/main.py#L178)

**5. Rich Metadata**
- Each chunk includes: `source_spec`, `method`, `path`, `chunk_id`
- Enables future metadata filtering and result organization
- **Implementation:** [`ai_exercise/main.py:78-90`](ai_exercise/main.py#L78-L90)

#### 📚 Future Improvements & Production Readiness
See [`improvements.md`](improvements.md) for detailed suggestions including:

**Retrieval Quality:**
- Re-ranking with cross-encoders (+10-20% MRR, +15-25% Precision@5)
- Hybrid search combining BM25 + semantic embeddings (+5-10% Recall@5)
- Query expansion with domain-specific synonyms (+5-10% Recall@5)
- Metadata filtering for service/method pre-filtering (+20-30% Precision)
- Upgraded embeddings model (text-embedding-3-large)

**Production Readiness:**
- Retry logic with exponential backoff
- Circuit breaker patterns
- Fallback LLM providers
- Rate limiting and caching
- Structured logging and monitoring
- Async operations for better performance
- Vector store scaling strategies

### Quick Start

```bash
# Install dependencies
make install

# Start the API
make dev-api

# In another terminal, load all 7 OpenAPI specs, this may take 30+ mins
curl http://localhost/load

# Run evaluation
make evaluate

# Or try the Streamlit frontend
make start-app
```