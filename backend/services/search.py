import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.services.embedding import embed_text
from backend.services.keyword_search import keyword_search
from backend.services.logger import log_query
from backend.services.metrics import record_query
from backend.services.query_complexity import query_complexity
from backend.services.query_expander import expand_query
from backend.services.query_rewriter import rewrite_query
from backend.services.rank_fusion import reciprocal_rank_fusion
from backend.services.reranker import rerank
from backend.services.router import detect_intent
from backend.services.vector_db import search_documents

logger = logging.getLogger(__name__)

TOP_K_COMPLEX: int = 20
TOP_K_SIMPLE: int = 8
RERANK_CANDIDATES_COMPLEX: int = 15
RERANK_CANDIDATES_SIMPLE: int = 6
FINAL_TOP_K: int = 3
KEYWORD_ONLY_TOP_K: int = 5
SCORE_THRESHOLD_VECTOR: float = 0.30
PARALLEL_WORKERS: int = 2


def search(
    query: str,
    workspace: str = "default",
    source: str | None = None,
) -> list[dict]:
    """
    Executes the search pipeline for a given query.
    """
    # Start Timer
    start: float = time.time()
    logger.info("Search started | workspace='%s' | query='%s'", workspace, query)

    # Intent Routing
    intent: str = detect_intent(query)
    logger.info("Intent: '%s' | query='%s'", intent, query)

    # ROUTING PATH A: Chat
    if intent == "chat":
        logger.info("Chat intent — skipping retrieval pipeline")
        latency_ms: float = (time.time() - start) * 1000
        record_query(latency_ms)
        log_query(query, [], latency_ms, workspace=workspace)
        return []

    # ROUTING PATH B: Keyword
    if intent == "keyword":
        logger.info("Keyword intent — running BM25 only | workspace='%s'", workspace)

        keyword_results: list[dict] = keyword_search(
            query,
            workspace=workspace,
            top_k=KEYWORD_ONLY_TOP_K,
        )

        unique_keyword_docs: list[dict] = []
        seen_keys: set[tuple] = set()

        for doc in keyword_results:
            doc_source: str = doc.get("source", "unknown")
            doc_chunk_id = doc.get("chunk_id", doc.get("text", ""))
            key: tuple = (doc_source, doc_chunk_id)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_keyword_docs.append(doc)

        final_keyword_docs: list[dict] = unique_keyword_docs[:FINAL_TOP_K]

        latency_ms = (time.time() - start) * 1000
        record_query(latency_ms)
        log_query(query, final_keyword_docs, latency_ms, workspace=workspace)
        logger.info(
            "Keyword search complete | results=%d | latency=%.2fms",
            len(final_keyword_docs),
            latency_ms,
        )
        return final_keyword_docs

    # ROUTING PATH C: Semantic — Full Hybrid Pipeline

    # PARALLEL Query Rewrite + Expand
    rewritten_query: str = query
    expanded_query: str = query

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures: dict = {
            executor.submit(rewrite_query, query): "rewrite",
            executor.submit(expand_query, query): "expand",
        }

        logger.info("Parallel LLM calls started: rewrite + expand")

        for future in as_completed(futures):
            label: str = futures[future]

            try:
                result: str = future.result()

                if label == "rewrite":
                    rewritten_query = result
                    logger.debug("Rewritten query ready: '%s'", rewritten_query)
                elif label == "expand":
                    expanded_query = result
                    logger.debug("Expanded query ready: '%s'", expanded_query)

            except Exception as e:
                logger.error(
                    "Parallel task '%s' failed: %s — using original query",
                    label,
                    e,
                )

    logger.info(
        "Parallel LLM calls complete | rewrite done | expand done"
    )

    # Query Complexity
    complexity: str = query_complexity(query)
    logger.info("Query complexity: '%s'", complexity)

    # Embed the Expanded Query
    query_vector: list[float] = embed_text(expanded_query)

    # Vector Search
    top_k: int = TOP_K_COMPLEX if complexity == "complex" else TOP_K_SIMPLE

    vector_results = search_documents(
        query_vector,
        workspace=workspace,
        limit=top_k,
        source=source,
    )

    vector_docs: list[dict] = [point.payload for point in vector_results]
    logger.info("Vector search returned %d results", len(vector_docs))

    # BM25 Keyword Search
    keyword_docs: list[dict] = keyword_search(expanded_query, workspace=workspace)
    logger.info("Keyword search returned %d results", len(keyword_docs))

    # Reciprocal Rank Fusion
    fused_docs: list[dict] = reciprocal_rank_fusion(vector_docs, keyword_docs)
    logger.info("After fusion: %d candidates", len(fused_docs))

    # Deduplication
    unique_docs: list[dict] = []
    seen_keys: set[tuple] = set()

    for doc in fused_docs:
        doc_source: str = doc.get("source", "unknown")
        doc_chunk_id: int | str = doc.get("chunk_id", doc.get("text", ""))
        key: tuple = (doc_source, doc_chunk_id)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_docs.append(doc)

    logger.info("After deduplication: %d unique candidates", len(unique_docs))

    # CrossEncoder Reranking
    if unique_docs:
        rerank_candidates_count: int = (
            RERANK_CANDIDATES_COMPLEX if complexity == "complex"
            else RERANK_CANDIDATES_SIMPLE
        )
        candidates: list[dict] = unique_docs[:rerank_candidates_count]
        ranked_docs: list[dict] = rerank(expanded_query, candidates, top_k=FINAL_TOP_K)
        logger.info("Reranking complete | top %d results selected", len(ranked_docs))
    else:
        ranked_docs = []
        logger.warning(
            "No results found | workspace='%s' | query='%s'",
            workspace,
            query,
        )

    final_docs: list[dict] = ranked_docs[:FINAL_TOP_K]

    # Metrics & Logging
    latency_ms: float = (time.time() - start) * 1000
    record_query(latency_ms)
    log_query(query, final_docs, latency_ms, workspace=workspace)

    logger.info(
        "Search complete | workspace='%s' | results=%d | latency=%.2fms",
        workspace,
        len(final_docs),
        latency_ms,
    )

    return final_docs