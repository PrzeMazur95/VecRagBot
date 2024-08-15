from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://qdrant:6333")
COLLECTION_NAME = "test_collection"

def create_collection() -> None:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=4, distance=Distance.DOT)
    )

def upser_sample_data_into_qdrant() -> None:
    client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ],
)

def check_if_collection_exists() -> bool:
    collections = client.get_collections()
    return COLLECTION_NAME in [collection.name for collection in collections.collections]

def main() -> None:
    if not check_if_collection_exists():
        create_collection()
        upser_sample_data_into_qdrant()

    search_result = client.search(
        collection_name=COLLECTION_NAME, query_vector=[0.19, 0.81, 0.75, 0.11], limit=1
    )

    print(search_result)

if __name__ == '__main__':
    main()
