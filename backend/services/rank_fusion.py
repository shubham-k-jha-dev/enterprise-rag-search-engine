def reciprocal_rank_fusion(vector_docs, keyword_docs, k=60):
    """
    Merge vector and keyword results using Reciprocal Rank Fusion.
    Each document gets a score of 1/(k + rank + 1) from each list.
    Documents appearing in both lists get scores from both added together.
    Higher combined score = more relevant.
    k=60 is the standard value used in research and production systems.
    """
    scores = {}

    def _process(docs, k):
        for rank, doc in enumerate(docs):
            # rank is the position of the doc in the list (0, 1, 2...)
            # doc is the dict with text, source, chunk_id
            text = doc.get("text", "")
            key = text  # use text as unique identity key

            score = 1 / (k + rank + 1)
            # higher rank (closer to 0) = higher score
            # k=60 prevents top results from dominating too much

            if key not in scores:
                # first time seeing this chunk — initialize with full metadata
                scores[key] = {
                    "score": 0,
                    "doc": {
                        "text": text,
                        "source": doc.get("source"),
                        "chunk_id": doc.get("chunk_id"),
                        "rerank_score": doc.get("rerank_score")
                    }
                }

            scores[key]["score"] += score
            # if chunk appeared in both vector and BM25, scores add up
            # making it rank higher in final results

    _process(vector_docs, k)
    _process(keyword_docs, k)

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    return [item["doc"] for item in ranked]