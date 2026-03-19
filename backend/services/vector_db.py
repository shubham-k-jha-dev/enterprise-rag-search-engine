import logging
import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()

# Logger :- 
logger = logging.getLogger(__name__)

# CONSTANTS
VECTOR_SIZE: int = 384
SCORE_THRESHOLD: float = 0.30
COLLECTION_PREFIX: str = "workspace_"
# workspace "got"   → collection "workspace_got"
# workspace "dexter"→ collection "workspace_dexter"
# The prefix prevents accidental name collisions with other data
QDRANT_MODE: str = os.getenv("QDRANT_MODE", "file").lower()
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_FILE_PATH: str = os.getenv("QDRANT_FILE_PATH", "data/qdrant")

if QDRANT_MODE == "server":
    client = QdrantClient(host = QDRANT_MODE, port = QDRANT_PORT)
    logger.info(
        "Qdrant client: SERVER mode | host = '%s' | port = %d",
        QDRANT_HOST,
        QDRANT_PORT
    )
else:
    client = QdrantClient(path = QDRANT_FILE_PATH)
    logger.info(
        "qdrant client: FILE mode | path = '%s'",
        QDRANT_FILE_PATH
    )

# Helper
def _collection_name(workspace: str) -> str:
    """ Returns the qdrant collection name for a given workspace."""
    return f"{COLLECTION_PREFIX}{workspace}"


# COLLECTION MANAGEMENT

def create_collection(workspace: str = "default") -> None:
    """
    Creates a Qdrant vector collection for the given workspace.
    """
    collection_name: str = _collection_name(workspace)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name = collection_name,
            vectors_config = VectorParams(
                size = VECTOR_SIZE,
                distance = Distance.COSINE
            ),
        )
        logger.info(
            "Qdrant collection created | collection = '%s'", collection_name
        )
    else:
        logger.debug(
            "Qdrant collection already exists | collection = '%s'", collection_name
        )

def delete_collection(workspace: str) -> bool:
    """
    Deletes the qdrant collection for the given workspace.
    """
    collection_name: str = _collection_name(workspace)
 
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            # delete_collection(name) permanently removes the collection
            # and ALL vectors stored in it.
            logger.info(
                "Qdrant collection deleted | collection='%s'", collection_name
            )
        else:
            logger.warning(
                "Delete requested but collection does not exist | collection='%s'",
                collection_name,
            )
        return True
 
    except Exception as e:
        logger.error(
            "Failed to delete Qdrant collection '%s': %s",
            collection_name,
            e,
        )
        return False


# DOCUMENT STORAGE

def store_document(
    vector: list[float],
    metadata: dict,
    workspace: str = "default",
) -> None:
    collection_name: str = _collection_name(workspace)
 
    if not client.collection_exists(collection_name):
        # Auto-create the collection if it doesn't exist. This handles dynamic workspaces — a new workspace created via POST /workspaces won't have a Qdrant collection until the first document is uploaded to it. We create it on-demand here.
        create_collection(workspace)
        logger.info(
            "Auto-created collection for workspace '%s' during store",
            workspace,
        )
 
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                # Each chunk gets a unique ID so we can upsert idempotently.
                vector=vector,
                payload=metadata,
            )
        ],
    )

# VECTOR SEARCH

def search_documents(
    query_vector: list[float],
    workspace: str = "default",
    limit: int = 10,
    source: str | None = None,
    score_threshold: float = SCORE_THRESHOLD,
):
    """
    Searches the workspace collection for the most similar vectors.
    """
    collection_name: str = _collection_name(workspace)
 
    query_filter = None
    # Default: no filter — search all documents in this collection.
 
    if source:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            ]
        )
 
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
    )
 
    filtered_points = [
        point for point in results.points
        if point.score >= score_threshold
    ]

    logger.info(
        "Vector search | workspace='%s' | mode='%s' | "
        "returned=%d | after_threshold=%d",
        workspace,
        QDRANT_MODE,
        len(results.points),
        len(filtered_points),
    )
 
    return filtered_points