def reciprocal_rank_fusion(vector_docs, keyword_docs, k=60):
    scores = {}

    # vector ranking
    for rank, doc in enumerate(vector_docs):
        raw = doc["text"]
        if isinstance(raw, dict):
            raw = raw.get("text", "")
        key = str(raw)

        score = 1 / (k + rank + 1)
        if key not in scores:
            scores[key] = {"score": 0, "doc": {"text": key, "source": doc.get("source", ""), "chunk_id": doc.get("chunk_id", -1)}}
        scores[key]["score"] += score

    # keyword ranking
    for rank, doc in enumerate(keyword_docs):
        if isinstance(doc, str):
            key = doc
        else:
            raw = doc.get("text", "")
            if isinstance(raw, dict):
                raw = raw.get("text", "")
            key = str(raw)

        score = 1 / (k + rank + 1)
        if key not in scores:
            scores[key] = {"score": 0, "doc": {"text": key}}
        scores[key]["score"] += score

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    return [item["doc"] for item in ranked]