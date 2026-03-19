import logging
import os
from rank_bm25 import BM25Okapi
from backend.services.chunking import semantic_chunk
from backend.services.document_processor import (
    _extract_text_from_txt,
    _extract_text_from_pdf,
    _extract_text_from_docx,
)
from backend.services.logger import get_workspace_files_from_db

# Logger 
logger = logging.getLogger(__name__)

# Constants :-
DATA_FOLDER: str = "data"
UPLOADS_FOLDER: str = "uploads"

# Workspace-aware BM25 indexes
# Structure: { "got": (bm25_index, [chunk_dicts]) }
_indexes: dict = {}

# Index Building :- 
def _get_chunks_for_workspace(workspace: str) -> list[dict]:
    """
    Loads and chunks all documents belonging to the given workspace.
    """
    registered_files: list[str] = get_workspace_files_from_db(workspace)
    if not registered_files:
        logger.warning(
            "No files registered for workspace '%s' in database", workspace
        )
        return []
    chunks: list[dict] = []

    for filename in registered_files:
        filepath: str | None = None

        for folder in [DATA_FOLDER, UPLOADS_FOLDER]:
            candidate: str = os.path.join(folder, filename)
            if os.path.exists(candidate):
                filepath = candidate
                break

        if filepath is None:
            # File is registered in SQLite but not found on disk.
            logger.warning(
                "Registered file not found on disk | filename = '%s' | workspace = '%s'",
                filename,
                workspace,
            )
            continue

        try:
            ext: str = os.path.splitext(filename)[1].lower()
            if ext == ".pdf":
                text: str = _extract_text_from_pdf(filepath)
            elif ext == ".docx":
                text: str = _extract_text_from_docx(filepath)
            else:
                text: str = _extract_text_from_txt(filepath)

            raw_chunks: list[str] = semantic_chunk(text)
            for i, chunk_text in enumerate(raw_chunks):
                chunks.append({
                    "text":     chunk_text,
                    "source":   filename,
                    "chunk_id": i,
                })
            logger.debug(
                "Loaded %d chunks from '%s' for workspace '%s'",
                len(raw_chunks),
                filename,
                workspace,
            )
        except Exception as e:
            logger.error(
                "Failed to read file '%s' for workspace '%s': %s",
                filename,
                workspace,
                e,
            )
            continue
    logger.info(
        "Loaded %d total chunks for workspace '%s' from %d files",
        len(chunks),
        workspace,
        len(registered_files),
    )
 
    return chunks



def _build_index(workspace: str) -> tuple:
    """
    Builds a BM23 index for the given workspace.
    """
    chunks: list[dict] = _get_chunks_for_workspace(workspace)
    if not chunks:
        logger.warning(
            "No chunks found for workspace '%s' — BM25 index not built",
            workspace,
        )
        return None, []
    tokenised: list[list[str]] = [
            chunk["text"].lower().split()
            for chunk in chunks
    ]    
    bm25_index = BM25Okapi(tokenised)
    logger.info(
        "BM25 index built for workspace '%s' | %d chunks indexed",
        workspace,
        len(chunks),
    )
 
    return bm25_index, chunks


def _ensure_index(workspace: str) -> None:
    """Build index for workspace if not already built."""
    if workspace not in _indexes:
        logger.info(
            "BM25 index not cached for workspace '%s' — building now",
            workspace,
        )
        bm25, chunks = _build_index(workspace)
        _indexes[workspace] = (bm25, chunks)


# PUBLIC API

def rebuild_bm25() -> None:
    """Clears all cached BM25 indexes.
        Called after a new document is uploaded via POST /upload.
        On the next search query, _ensure_index() will rebuild the index
        from disk, including the newly uploaded document.
    """
    global _indexes
    _indexes = {}
    logger.info("BM25 indexes cleared — will rebuild on next query")


def keyword_search(
    query: str,
    workspace: str = "default",
    top_k: int = 5,
    score_threshold: float = -1.0,
) -> list[dict]:
    """
    Runs BM25 keyword search for the given query in the given workspace.
    """
    _ensure_index(workspace)

    bm25, documents = _indexes.get(workspace, (None, []))

    if not bm25 or not documents:
        logger.warning(
            "No BM25 index available for workspace '%s' — returning empty results",
            workspace,
        )
        return []

    tokenized_query: list[str] = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked: list[tuple] = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results: list[dict] = []

    for doc, score in ranked:
        logger.debug("BM25 score: %.4f | source='%s'", score, doc.get("source"))
 
        if score < score_threshold:
            break 
        results.append(doc)
 
        if len(results) >= top_k:
            break
 
    logger.info(
        "BM25 search complete | workspace='%s' | results=%d",
        workspace,
        len(results),
    )
 
    return results