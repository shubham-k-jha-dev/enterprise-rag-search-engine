from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_k=3):

    pairs = [(query, doc) for doc in docs]

    scores = model.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked = [doc for doc, score in scored_docs[:top_k]]

    return reranked