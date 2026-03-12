# AI Search Engine

A production-grade **Retrieval-Augmented Generation (RAG)** search engine built with FastAPI, featuring agentic retrieval, hybrid search, cross-encoder reranking, and real-time streaming answers powered by large language models.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Search Pipeline](#search-pipeline)
- [Document Ingestion](#document-ingestion)
- [Evaluation](#evaluation)

---

## Overview

This project implements an end-to-end AI-powered search engine that goes beyond simple keyword matching. It combines **semantic vector search** with **BM25 keyword search**, fuses results using **Reciprocal Rank Fusion (RRF)**, and applies **cross-encoder reranking** to surface the most relevant documents. An **agentic retrieval** layer decomposes complex queries into subqueries for comprehensive coverage. Final answers are generated via an LLM and **streamed in real-time** to the client.

The system supports **multi-workspace isolation**, allowing separate document collections to be queried independently — useful for domain-specific knowledge bases.

---

## Architecture

```
                         +------------------+
                         |   Client / API   |
                         +--------+---------+
                                  |
                         +--------v---------+
                         |  FastAPI Server   |
                         |  (Rate Limited)   |
                         +--------+---------+
                                  |
              +-------------------+-------------------+
              |                                       |
     +--------v---------+                   +---------v--------+
     |  /search Endpoint |                   |  /ask Endpoint   |
     +--------+---------+                   +---------+--------+
              |                                       |
              v                                       v
     +------------------+                   +-------------------+
     | Search Pipeline  |                   | Agentic Retrieval |
     |                  |                   |  (Subquery Plan)  |
     | 1. Query Rewrite |                   +--------+----------+
     | 2. Query Expand  |                            |
     | 3. Complexity    |                   +--------v----------+
     |    Detection     |                   |  Search Pipeline  |
     | 4. Vector Search |                   |  (per subquery)   |
     | 5. BM25 Search   |                   +--------+----------+
     | 6. RRF Fusion    |                            |
     | 7. Deduplication |                   +--------v----------+
     | 8. Cross-Encoder |                   |  Deduplication    |
     |    Reranking     |                   +--------+----------+
     +--------+---------+                            |
              |                             +--------v----------+
              v                             | Answer Generator  |
     +------------------+                   | (LLM Streaming)   |
     |  Ranked Results  |                   +--------+----------+
     +------------------+                            |
                                            +--------v----------+
                                            | Streamed Response |
                                            +-------------------+
              
     +-------------------+     +-------------------+
     |    Qdrant          |     |      Redis        |
     | (Vector Database)  |     |  (Response Cache) |
     +-------------------+     +-------------------+
```

---

## Key Features

| Feature | Description |
|---|---|
| **Hybrid Search** | Combines dense vector retrieval (Qdrant) with sparse keyword search (BM25) for high recall |
| **Reciprocal Rank Fusion** | Merges vector and keyword rankings into a unified relevance score |
| **Cross-Encoder Reranking** | Applies `ms-marco-MiniLM-L-6-v2` cross-encoder for fine-grained relevance scoring |
| **Query Rewriting** | LLM-powered reformulation of user queries for improved clarity |
| **Query Expansion** | Augments queries with related keywords and concepts via LLM |
| **Query Complexity Detection** | Classifies queries as simple or complex to adjust retrieval depth |
| **Agentic Retrieval** | Decomposes multi-part queries into subqueries and aggregates results |
| **Streaming Answers** | Real-time token-by-token LLM response streaming via `StreamingResponse` |
| **Redis Caching** | Caches LLM-generated answers with 1-hour TTL to reduce latency and API costs |
| **Workspace Isolation** | Supports multiple independent document collections |
| **Document Upload** | Accepts `.txt`, `.pdf`, and `.docx` files via API with automatic chunking and indexing |
| **Semantic Chunking** | Sentence-boundary-aware chunking with configurable overlap for context preservation |
| **Rate Limiting** | Per-IP request throttling (30 requests/minute) via SlowAPI |
| **Observability** | Query logging, latency tracking, and cache hit rate metrics |
| **Intent Routing** | Classifies queries into semantic, keyword, or conversational intents |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI + Uvicorn |
| **LLM Provider** | Groq (Llama 3.3 70B Versatile) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Vector Database** | Qdrant (cosine similarity) |
| **Cache** | Redis |
| **Keyword Search** | BM25 via `rank-bm25` |
| **Document Parsing** | PyPDF, python-docx |
| **Containerization** | Docker + Docker Compose |
| **Language** | Python 3.11 |

---

## Project Structure

```
ai-search-engine/
├── backend/
│   ├── main.py                          # FastAPI app, endpoints, lifespan
│   ├── __init__.py
│   └── services/
│       ├── agentic_retrieval.py         # Subquery decomposition and aggregation
│       ├── answer_generator.py          # LLM streaming answer with source attribution
│       ├── cache.py                     # Redis caching layer
│       ├── chunking.py                  # Semantic sentence-based chunking
│       ├── document_processor.py        # TXT/PDF/DOCX text extraction and indexing
│       ├── embedding.py                 # Sentence transformer embeddings
│       ├── keyword_search.py            # BM25 keyword search with workspace filtering
│       ├── logger.py                    # JSON query logging
│       ├── metrics.py                   # Query count, latency, cache hit tracking
│       ├── query_complexity.py          # Simple/complex query classification
│       ├── query_expander.py            # LLM-powered query expansion
│       ├── query_planner.py             # Workspace routing for comparisons
│       ├── query_rewriter.py            # LLM-powered query reformulation
│       ├── rag.py                       # Basic RAG pipeline (search + generate)
│       ├── rank_fusion.py               # Reciprocal Rank Fusion implementation
│       ├── reranker.py                  # Cross-encoder reranking
│       ├── router.py                    # Intent detection (semantic/keyword/chat)
│       ├── search.py                    # Core hybrid search orchestrator
│       └── vector_db.py                 # Qdrant client operations
├── scripts/
│   ├── index_documents.py              # Batch indexing from a folder
│   ├── ingest_data.py                  # Single file ingestion
│   ├── evaluate_retrieval.py           # Offline retrieval evaluation
│   └── evaluate_live_retrieval.py      # Live retrieval evaluation
├── data/                                # Document store and Qdrant persistence
├── uploads/                             # Uploaded document storage
├── frontend/                            # Frontend application (placeholder)
├── docker/                              # Additional Docker configurations
├── .env                                 # Environment variables (API keys)
├── Dockerfile                           # Container image definition
├── docker-compose.yml                   # Multi-service orchestration
└── requirements.txt                     # Python dependencies
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose** (for containerized deployment)
- **Groq API Key** (for LLM-powered query processing and answer generation)

### Environment Variables

Create a `.env` file in the project root with the following:

```env
GROQ_API_KEY=your_groq_api_key_here
REDIS_HOST=localhost
```

> When running via Docker Compose, set `REDIS_HOST=redis` to use the containerized Redis instance.

### Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/shubham-k-jha-dev/ai-search-engine.git
   cd ai-search-engine
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start Redis** (required for caching)

   ```bash
   docker run -d -p 6379:6379 redis
   ```

5. **Index documents** (optional — load sample data)

   ```bash
   python scripts/index_documents.py data/
   ```

6. **Start the server**

   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

   The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Docker Deployment

Launch all services (backend, Redis, Qdrant) with a single command:

```bash
docker-compose up --build
```

This starts:
- **Backend** on port `8000`
- **Redis** on port `6379`
- **Qdrant** on port `6333` (with persistent storage at `./data/qdrant`)

---

## API Reference

### Health Check

```
GET /health
```

Returns service status. Used by Docker for health checks.

**Response:**
```json
{ "status": "ok", "service": "ai-search-engine" }
```

---

### Search

```
POST /search
```

Executes the full hybrid search pipeline and returns ranked document chunks.

**Request Body:**
```json
{
  "query": "What is the Iron Throne?",
  "workspace": "got",
  "source": null
}
```

**Response:**
```json
{
  "query": "What is the Iron Throne?",
  "workspace": "got",
  "results": [
    {
      "text": "The Iron Throne is the seat of the king...",
      "source": "got.txt",
      "chunk_id": 3,
      "rerank_score": 0.92
    }
  ]
}
```

---

### Ask (Streaming Answer)

```
GET /ask?query=Who is Dexter Morgan?&workspace=dexter
```

Runs agentic retrieval, then streams an LLM-generated answer token-by-token with source attribution appended at the end.

**Response:** `text/plain` streaming response.

---

### Upload Document

```
POST /upload?workspace=default
```

Upload a `.txt`, `.pdf`, or `.docx` file. The document is automatically chunked, embedded, and indexed into the specified workspace.

**Response:**
```json
{
  "message": "Document uploaded and indexed successfully",
  "file": "report.pdf",
  "workspace": "default",
  "chunks_created": 42
}
```

---

### List Workspaces

```
GET /workspaces
```

Returns all available workspaces.

**Response:**
```json
{ "workspaces": ["default", "got", "dexter"] }
```

---

### Metrics

```
GET /metrics
```

Returns runtime performance metrics.

**Response:**
```json
{
  "total_queries": 150,
  "avg_latency_ms": 342.67,
  "cache_hit_rate": 0.213
}
```

---

## Search Pipeline

The search pipeline processes every query through the following stages:

1. **Query Rewriting** — The raw query is sent to the Groq LLM to produce a clearer, more specific reformulation.

2. **Query Expansion** — The query is augmented with semantically related terms and concepts via the LLM.

3. **Complexity Classification** — Queries are classified as `simple` or `complex` based on linguistic patterns (comparison keywords, analytical terms, query length). Complex queries retrieve more documents (top-20 vs top-8).

4. **Vector Search** — The expanded query is embedded using `all-MiniLM-L6-v2` and searched against Qdrant using cosine similarity. A minimum score threshold of `0.30` filters low-relevance results.

5. **BM25 Keyword Search** — A parallel sparse retrieval pass using the BM25 algorithm over tokenized document chunks, with workspace-scoped indexing.

6. **Reciprocal Rank Fusion** — Vector and keyword results are merged using RRF (k=60), producing a single ranked list that balances both retrieval signals.

7. **Deduplication** — Duplicate chunks (identified by source file + chunk ID) are removed.

8. **Cross-Encoder Reranking** — The fused results are re-scored using a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) for pairwise relevance estimation. The top 3 results are returned.

---

## Document Ingestion

Documents can be ingested through two methods:

### Via API Upload

Upload files through the `/upload` endpoint. Supported formats: `.txt`, `.pdf`, `.docx`.

```bash
curl -X POST "http://localhost:8000/upload?workspace=default" \
  -F "file=@document.pdf"
```

### Via CLI Scripts

**Single file:**
```bash
python scripts/ingest_data.py path/to/document.txt
```

**Entire folder:**
```bash
python scripts/index_documents.py path/to/folder/
```

### Processing Pipeline

1. **Text Extraction** — Raw text is extracted from the file using format-specific parsers (PyPDF for PDFs, python-docx for DOCX files).
2. **Semantic Chunking** — Text is split at sentence boundaries into chunks of 3 sentences with 1-sentence overlap to maintain cross-chunk context.
3. **Embedding** — Each chunk is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`.
4. **Storage** — Vectors and metadata (text, source filename, chunk ID) are upserted into the workspace-specific Qdrant collection.

---

## Evaluation

The `scripts/` directory includes evaluation utilities:

- **`evaluate_retrieval.py`** — Runs offline evaluation against a test dataset (`data/evaluation.json`).
- **`evaluate_live_retrieval.py`** — Evaluates retrieval quality against the live running system.

---

## License

This project is developed for educational and demonstration purposes.
