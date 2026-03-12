from backend.services.vector_db import search_documents
from backend.services.embedding import embed_text
from backend.services.reranker import rerank
import time
from backend.services.logger import log_query
from backend.services.keyword_search import keyword_search
from backend.services.metrics import record_query
from backend.services.query_expander import expand_query
from backend.services.rank_fusion import reciprocal_rank_fusion
from backend.services.query_complexity import query_complexity
from backend.services.query_rewriter import rewrite_query


def search(query: str, workspace="default", source=None):

    start = time.time()

    # QUERY REWRITING
    rewritten_query = rewrite_query(query)
    print(f"REWRITTEN QUERY: {rewritten_query}")
    
    # QUERY EXPANSION
    expanded_query = expand_query(query)
    print(f"EXPANDED QUERY: {expanded_query}")
    
    complexity = query_complexity(query)
    print("QUERY COMPLEXITY:", complexity)


    # VECTOR SEARCH
    query_vector = embed_text(expanded_query)

    if complexity == "complex":
        top_k = 20
    else:
        top_k = 8

    vector_results = search_documents(query_vector, workspace=workspace, limit=top_k, source=source)
    
    for r in vector_results:
        print("VECTOR SCORE:", r.score)

    vector_docs = [r.payload for r in vector_results]

    print("VECTOR RESULTS:", vector_docs)


    # KEYWORD SEARCH (BM25)
    keyword_docs = keyword_search(expanded_query, workspace=workspace)
    print("KEYWORD RESULTS:", keyword_docs)


    # HYBRID FUSION
    fused_docs = reciprocal_rank_fusion(vector_docs, keyword_docs)
    print("FUSED RESULTS:", fused_docs)


    # REMOVE DUPLICATES (VERY IMPORTANT)
    unique_docs = []
    seen = set()

    for doc in fused_docs:
        source = doc.get("source", "unknown")
        chunk_id = doc.get("chunk_id", doc.get("text", ""))
        key = (source, chunk_id)
        if key not in seen:
            unique_docs.append(doc)
            seen.add(key)

    fused_docs = unique_docs

    # RERANK WITH CROSS ENCODER
    if fused_docs:

        if complexity == "complex":
            texts = [doc["text"] for doc in fused_docs][:15]
        else:
            texts = [doc["text"] for doc in fused_docs][:6]

        ranked_texts = rerank(expanded_query, texts)

        ranked_docs = []

        for text in ranked_texts:
            for doc in fused_docs:
                if doc["text"] == text:
                    ranked_docs.append(doc)
                    break

    else:
        ranked_docs = []

    # FINAL TOP RESULTS
    final_docs = ranked_docs[:3]

    # METRICS + LOGGING
    latency = (time.time() - start) * 1000

    record_query(latency)

    log_query(query, final_docs, latency)


    return final_docs