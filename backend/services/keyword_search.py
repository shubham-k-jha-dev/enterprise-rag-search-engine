from rank_bm25 import BM25Okapi
import os

DATA_FOLDER = "data"
UPLOADS_FOLDER = "uploads"

# Workspace-aware BM25 indexes
_indexes = {}

def _get_workspace_files(workspace: str):
    """Return all chunks belonging to a specific workspace based on filename."""
    chunks = []
    for folder in [DATA_FOLDER, UPLOADS_FOLDER]:
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            if not filename.endswith(".txt"):
                continue
            if not filename.startswith(workspace):
                continue
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding = "utf-8") as f:
                text = f.read()
                for chunk in text.split("\n\n"):
                    chunk = chunk.strip()
                    if chunk:
                        chunks.append(chunk)
    
    return chunks
    
def _build_index(workspace: str):
    chunks = _get_workspace_files(workspace)
    if not chunks:
        return None, []
    tokenized = [doc.lower().split() for doc in chunks]
    return BM25Okapi(tokenized), chunks


def _ensure_index(workspace: str):
    """Build index for workspace if not already built."""
    if workspace not in _indexes:
        bm25, chunks = _build_index(workspace)
        _indexes[workspace] = (bm25, chunks)
        

def rebuild_bm25():
    """Rebuild all workspace indexes. Call after upload."""
    global _indexes
    _indexes = {}
    print("BM25 indexes cleared — will rebuild on next query")


def keyword_search(
    query: str, 
    workspace: str = "default", 
    top_k: int = 5, 
    score_threshold: float = 0.5
):
    _ensure_index(workspace)

    bm25, documents = _indexes.get(workspace, (None, []))

    if not bm25 or not documents:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for doc, score in ranked:
        print("BM25 SCORE:", score)
        if score >= score_threshold:
            results.append(doc)

        if len(results) >= top_k:
            break

    return results