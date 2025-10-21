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

## Assignment Submission

### What I Implemented

#### 1. Multi-Spec Knowledge Base
The RAG system now loads all 7 StackOne OpenAPI specs (stackone, hris, ats, lms, iam, crm, marketing) via the `/load` endpoint.

#### 2. Evaluation Framework
- Built Ragas-based evaluation with 32 test queries
- Measures 4 key metrics: context precision, recall, faithfulness, answer relevancy
- Run with: `make evaluate`
- See [evaluation_guide.md](evaluation_guide.md) for details

#### 3. Retrieval Improvements Implemented
- **Contextual Retrieval**: Each chunk is contextualized using LLM before embedding
- **Query Rewriting**: User queries are rewritten for better semantic matching
- **Smart Chunking**: Separate chunks for endpoints, schemas, and security definitions
- **Distance Filtering**: Filters out irrelevant results based on embedding distance

#### 4. Future Improvements
See [improvements.md](improvements.md) for detailed suggestions on:
- Re-ranking with cross-encoders
- Hybrid search (BM25 + semantic)
- Production readiness (retry logic, caching, monitoring, etc.)