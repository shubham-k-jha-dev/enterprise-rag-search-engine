# AI Search Engine

A production-grade agentic retrieval system that combines vector search, BM25 keyword search, reciprocal rank fusion, cross-encoder reranking, and real-time LLM streaming to deliver accurate, context-grounded answers from document collections.

Built with FastAPI, Qdrant, Redis, Groq (Llama 3.3 70B), and a React + TypeScript frontend.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Search Pipeline](#search-pipeline)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Evaluation Framework](#evaluation-framework)
- [Testing](#testing)

---

## Architecture Overview

```
Client Request
      |
      v
+------------------+
|   FastAPI Gateway |  (main.py)
|   - Auth (API Key)|
|   - Rate Limiting |
|   - CORS          |
+--------+---------+
         |
         v
+------------------+     +-------------------+
| Intent Router    |---->| Chat: return []   |
| (router.py)      |---->| Keyword: BM25 only|
+--------+---------+---->| Semantic: full    |
         |                +-------------------+
         v
+------------------+
| Agentic Planner  |  (query_planner.py + agentic_retrieval.py)
| - Sub-queries    |
| - Cross-workspace|
+--------+---------+
         |
         v (per sub-query)
+------------------+
| Search Pipeline  |  (search.py)
| 1. Rewrite query |  (query_rewriter.py)  --|
| 2. Expand query  |  (query_expander.py)  --|--> Parallel via ThreadPoolExecutor
| 3. Complexity    |  (query_complexity.py)
| 4. Embed         |  (embedding.py — all-MiniLM-L6-v2)
| 5. Vector search |  (vector_db.py — Qdrant)
| 6. BM25 search   |  (keyword_search.py — rank-bm25)
| 7. Rank fusion   |  (rank_fusion.py — RRF k=60)
| 8. Deduplicate   |
| 9. Rerank        |  (reranker.py — ms-marco-MiniLM-L-6-v2)
+--------+---------+
         |
         v
+------------------+
| Answer Generator |  (answer_generator.py)
| - Redis cache    |  (cache.py)
| - Conv. history  |  (conversation.py)
| - Groq streaming |  (llama-3.3-70b-versatile)
| - Source attrib.  |
+------------------+
         |
         v
   Streamed Response
```

---

## Search Pipeline

The retrieval pipeline implements a multi-stage approach to maximize relevance:

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Intent Routing** | Classifies queries as `chat` (skip retrieval), `keyword` (BM25 only), or `semantic` (full pipeline) |
| 2 | **Query Rewriting** | LLM rewrites the query for better retrieval (Groq API) |
| 3 | **Query Expansion** | LLM expands the query with related terms and synonyms (Groq API) |
| 4 | **Parallel Execution** | Steps 2 and 3 run concurrently via `ThreadPoolExecutor`, reducing latency from ~1200ms to ~600ms |
| 5 | **Query Complexity** | Rule-based classifier determines if the query is `simple` or `complex`, adjusting top-k and rerank candidate counts |
| 6 | **Embedding** | The expanded query is embedded using `all-MiniLM-L6-v2` (384-dimensional vectors) |
| 7 | **Vector Search** | Top-k nearest neighbors from Qdrant (cosine similarity, threshold 0.30) |
| 8 | **Keyword Search** | BM25 lexical search over the same workspace (score threshold 0.50) |
| 9 | **Reciprocal Rank Fusion** | Merges vector and keyword results using RRF (k=60) so documents appearing in both lists are boosted |
| 10 | **Deduplication** | Removes duplicate chunks by (source, chunk_id) |
| 11 | **Cross-Encoder Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all (query, chunk) pairs and selects the top 3 |

---

## Features

### Hybrid Search
- Combines dense vector retrieval (semantic understanding) with sparse BM25 retrieval (exact keyword matching)
- Reciprocal rank fusion ensures documents relevant by either method are surfaced

### Agentic Retrieval
- Automatic sub-query decomposition: queries containing "and" are split into independent sub-queries
- "Why" queries are augmented with "reason for" variants
- Cross-workspace comparison queries are detected and routed to all non-default workspaces simultaneously

### Dynamic Workspaces
- Create and delete workspaces at runtime via the API
- Each workspace maps to an independent Qdrant collection and BM25 index
- Default workspaces (`default`, `got`, `dexter`) are protected from deletion
- Workspace registry persisted in SQLite across server restarts

### Multi-Turn Conversation Memory
- Each browser session generates a unique conversation ID (UUID v4)
- Conversation history is stored in SQLite and loaded into the LLM prompt
- The LLM maintains context across follow-up questions within the same session

### Real-Time LLM Streaming
- Answers are streamed token-by-token from Groq (Llama 3.3 70B)
- Graceful fallback: if the LLM is unavailable, raw retrieved chunks are returned
- Source attribution appended to every response

### Redis Caching
- Completed answers are cached in Redis with a 1-hour TTL
- Subsequent identical queries return instantly from cache

### Document Ingestion
- Supports `.txt`, `.pdf`, and `.docx` file formats
- Semantic chunking with 3-sentence windows and 1-sentence overlap
- Each chunk is embedded and stored in Qdrant with full metadata
- Document sources are registered in SQLite for keyword search discovery

### Observability
- Structured logging with `logging` module throughout all services
- Query latency tracking and cache hit rate metrics
- Query logs persisted to SQLite with timestamps, results, and workspace context

### Security
- API key authentication via `X-API-Key` header on all mutating endpoints
- Rate limiting at 30 requests/minute per IP address (SlowAPI)
- CORS configured for frontend origin

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Framework** | FastAPI 0.135 | HTTP server, request validation, OpenAPI docs |
| **Vector Database** | Qdrant (local/Docker) | Dense vector storage and similarity search |
| **Cache** | Redis | Response caching (1-hour TTL) |
| **Relational DB** | SQLite | Query logs, conversations, workspace registry, document sources |
| **Embedding Model** | `all-MiniLM-L6-v2` | 384-dimensional sentence embeddings (sentence-transformers) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking of candidate documents |
| **LLM** | Llama 3.3 70B via Groq API | Answer generation with streaming |
| **Keyword Search** | rank-bm25 | BM25Okapi implementation for lexical retrieval |
| **Frontend** | React 18 + TypeScript + Vite | Single-page application with tabs for all operations |
| **Containerization** | Docker + Docker Compose | Multi-service deployment (backend, Redis, Qdrant) |
| **Testing** | pytest + FastAPI TestClient | API endpoint integration tests |

---

## Project Structure

```
ai-search-engine/
|
|-- backend/
|   |-- main.py                    # FastAPI application, routes, middleware, lifespan
|   |-- services/
|       |-- search.py              # Master search orchestrator with parallel LLM calls
|       |-- agentic_retrieval.py   # Sub-query planning and multi-workspace execution
|       |-- query_planner.py       # Comparison vs single query routing
|       |-- query_rewriter.py      # LLM-based query rewriting (Groq)
|       |-- query_expander.py      # LLM-based query expansion (Groq)
|       |-- query_complexity.py    # Rule-based complexity classifier
|       |-- router.py              # Intent detection (chat/keyword/semantic)
|       |-- embedding.py           # Sentence embedding (all-MiniLM-L6-v2)
|       |-- vector_db.py           # Qdrant collection and search operations
|       |-- keyword_search.py      # BM25 index management and search
|       |-- rank_fusion.py         # Reciprocal rank fusion (RRF)
|       |-- reranker.py            # Cross-encoder reranking
|       |-- answer_generator.py    # Groq LLM streaming with fallback
|       |-- cache.py               # Redis get/set with TTL
|       |-- conversation.py        # Multi-turn history load/store
|       |-- document_processor.py  # File extraction, chunking, embedding, indexing
|       |-- chunking.py            # Semantic sentence-window chunking
|       |-- logger.py              # SQLite: query logs, conversations, workspaces
|       |-- metrics.py             # In-memory query performance counters
|
|-- frontend/
|   |-- src/pages/Index.tsx        # Main UI: Search, Ask AI, Upload, Workspaces, Metrics
|   |-- index.html                 # Entry point
|   |-- vite.config.ts             # Vite build configuration
|
|-- scripts/
|   |-- index_documents.py         # Batch document indexing from a folder
|   |-- ingest_data.py             # Data ingestion utility
|   |-- evaluate_retrieval.py      # Recall@k, MRR evaluation against ground truth
|   |-- evaluate_answers.py        # LLM answer hit rate and faithfulness evaluation
|   |-- evaluate_live_retrieval.py # Live retrieval testing
|
|-- tests/
|   |-- test_api.py                # pytest: health, auth, search, workspace, logs
|
|-- data/
|   |-- evaluation.json            # 20 ground-truth test cases (10 GoT + 10 Dexter)
|   |-- got.txt                    # Game of Thrones source document
|   |-- dexter.txt                 # Dexter source document
|   |-- qdrant/                    # Qdrant persistent storage
|   |-- query_logs.db              # SQLite database
|
|-- docker/                        # Docker-related configuration
|-- Dockerfile                     # Python 3.11.9 backend image
|-- docker-compose.yml             # Redis + Qdrant + backend orchestration
|-- requirements.txt               # Python dependencies
|-- .env                           # Environment variables (not committed)
```

---

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher (for the frontend)
- Redis server (local or Docker)
- Groq API key ([console.groq.com](https://console.groq.com))

### Backend Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-search-engine

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Initial Data Indexing

Before searching, documents must be indexed into Qdrant:

```bash
# Index documents from the data/ folder
python scripts/index_documents.py data/

# Or use the ingest script
python scripts/ingest_data.py
```

Alternatively, upload documents through the frontend Upload tab or the `POST /upload` endpoint after the server is running.

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
API_KEY=your_api_key_here
REDIS_HOST=localhost
FRONTEND_ORIGIN=http://localhost:8080
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key for Groq LLM service (query rewriting, expansion, answer generation) |
| `API_KEY` | Yes | API key clients must send in the `X-API-Key` header |
| `REDIS_HOST` | No | Redis hostname (default: `localhost`) |
| `FRONTEND_ORIGIN` | No | Allowed CORS origin (default: `http://localhost:8080`) |

For the frontend, set `VITE_API_KEY` in `frontend/.env`:

```env
VITE_API_KEY=your_api_key_here
```

---

## Running the Application

### Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 redis

# Or install locally and run
redis-server
```

### Start the Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The API documentation is available at `http://localhost:8000/docs` (Swagger UI).

### Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend runs at `http://localhost:8080` by default.

---

## Docker Deployment

The included `docker-compose.yml` orchestrates all three services:

```bash
docker-compose up --build
```

This starts:

| Service | Port | Description |
|---------|------|-------------|
| `backend` | 8000 | FastAPI application |
| `redis` | 6379 | Response cache |
| `qdrant` | 6333 | Vector database with persistent storage at `./data/qdrant` |

---

## API Reference

All mutating endpoints require the `X-API-Key` header.

### System

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | No | Health check |
| `GET` | `/metrics` | No | Query performance metrics (total queries, avg latency, cache hit rate) |
| `GET` | `/logs` | Yes | 20 most recent query log entries |

### Workspaces

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/workspaces` | No | List all registered workspaces |
| `POST` | `/workspaces` | Yes | Create a new workspace (name: lowercase, digits, hyphens, 1-50 chars) |
| `DELETE` | `/workspaces/{name}` | Yes | Delete a workspace and its Qdrant collection (defaults are protected) |

### Retrieval

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/search` | Yes | Hybrid search with reranking. Body: `{"query": "...", "workspace": "...", "source": null}` |
| `GET` | `/ask` | Yes | Agentic retrieval + streaming LLM answer. Params: `query`, `workspace`, `conversation_id` |

### Ingestion

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/upload` | Yes | Upload and index a document (.txt, .pdf, .docx). Params: `workspace`. Body: multipart file |

### Response Formats

**POST /search** response:
```json
{
  "query": "Who killed Ned Stark?",
  "workspace": "got",
  "results": [
    {
      "text": "Ned Stark was executed on the orders of King Joffrey...",
      "source": "got.txt",
      "chunk_id": 3,
      "rerank_score": 0.9847
    }
  ]
}
```

**GET /ask** response: Streamed plain text with source attribution appended at the end.

---

## Evaluation Framework

The project includes two evaluation scripts that measure retrieval and answer quality against a ground-truth dataset of 20 test cases (`data/evaluation.json`).

### Retrieval Evaluation

Measures whether the correct document chunk is retrieved in the top-k results.

```bash
python scripts/evaluate_retrieval.py --workspace got
python scripts/evaluate_retrieval.py --workspace dexter --top_k 5
```

**Metrics computed:**
- **Recall@k**: Fraction of queries where the expected chunk was found in the top-k results
- **MRR (Mean Reciprocal Rank)**: Average of `1/rank` for each query (1.0 = always at rank 1)
- **Per-query-type breakdown**: Separate metrics for factoid, reasoning, and event queries
- **Latency statistics**: Average, minimum, and maximum per-query latency

Results are saved to `data/eval_results_{workspace}.json`.

### Answer Evaluation

Measures whether the LLM-generated answer contains the expected information. Requires the server to be running.

```bash
python scripts/evaluate_answers.py --workspace got
python scripts/evaluate_answers.py --workspace dexter --base_url http://localhost:8000
```

**Metrics computed:**
- **Answer Hit Rate**: Fraction of answers containing the expected text
- **Faithfulness Rate**: Fraction of answers that appear to use the retrieved context (not refusing or hallucinating)
- **Average Answer Length**: Mean character count of generated answers

Results are saved to `data/answer_eval_{workspace}.json`.

---

## Testing

Run the test suite with pytest:

```bash
pytest tests/ -v
```

**Test coverage includes:**
- Health check endpoint
- Workspace listing (verifies default workspaces present)
- Authentication enforcement (missing key, wrong key)
- Search response shape and structure validation
- Workspace isolation (Dexter queries only return Dexter sources)
- Query log retrieval

---

## Key Design Decisions

1. **Hybrid search over vector-only**: Pure vector search misses exact keyword matches. BM25 complements semantic search for queries with specific names or terms.

2. **ThreadPoolExecutor for parallel LLM calls**: Query rewriting and expansion are I/O-bound (waiting for Groq API responses). Running them concurrently halves the preprocessing latency without requiring an async rewrite.

3. **Cross-encoder reranking as final stage**: Bi-encoder retrieval is fast but approximate. The cross-encoder jointly scores each (query, document) pair for precise relevance ranking at the cost of being slower — applied only to the top candidates.

4. **Reciprocal Rank Fusion (k=60)**: RRF is parameter-light and robust. Documents appearing in both vector and keyword results receive boosted scores, surfacing the most comprehensively relevant results.

5. **SQLite for workspace and document registry**: Lightweight, zero-configuration persistence. Workspace membership and document source tracking survive server restarts without requiring a separate database service.

6. **Streaming responses**: Token-by-token streaming provides immediate user feedback. The frontend renders the answer progressively as it arrives.

7. **Graceful LLM fallback**: If the Groq API is unreachable or returns an error, the system returns the raw retrieved chunks instead of failing entirely.
