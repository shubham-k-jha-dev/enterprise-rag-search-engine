from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid

client = QdrantClient(path="data/qdrant")

def create_collection(workspace="default"):
    collection_name = f"workspace_{workspace}"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

def store_document(vector, metadata, workspace="default"):
    collection_name = f"workspace_{workspace}"
    if not client.collection_exists(collection_name):
        create_collection(workspace)
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=metadata
            )
        ]
    )

def search_documents(query_vector, workspace="default", limit=10, source=None, score_threshold=0.30):
    collection_name = f"workspace_{workspace}"
    query_filter = None

    if source:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )
            ]
        )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter
    )

    filtered_points = []

    for point in results.points:
        print("VECTOR SCORE:", point.score)
        if point.score >= score_threshold:
            filtered_points.append(point)

    return filtered_points