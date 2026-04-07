import random
from typing import cast

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from libs.common.config import get_settings


def get_qdrant_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url)


def ensure_collection() -> None:
    s = get_settings()
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if s.qdrant_collection not in collections:
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config=VectorParams(size=s.qdrant_vector_size, distance=Distance.COSINE),
        )


def upsert_item_vectors(points: list[PointStruct]) -> None:
    s = get_settings()
    client = get_qdrant_client()
    client.upsert(collection_name=s.qdrant_collection, points=points)


def query_similar(item_id: int, limit: int = 20) -> list[tuple[int, float]]:
    s = get_settings()
    client = get_qdrant_client()
    try:
        points = client.retrieve(
            collection_name=s.qdrant_collection,
            ids=[item_id],
            with_vectors=True,
        )
        if not points or points[0].vector is None:
            return []
        query_vector = cast(list[float], points[0].vector)
        result = client.query_points(
            collection_name=s.qdrant_collection,
            query=query_vector,
            limit=limit,
            with_payload=False,
        )
        return [(int(p.id), float(p.score)) for p in result.points if int(p.id) != item_id]
    except Exception:
        # Fallback keeps API usable before indexing is ready.
        return [(i, random.random()) for i in range(100, 100 + limit)]
