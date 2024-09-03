"""
Model to interact with Qdrant vector database
"""
import os

from qdrant_client import QdrantClient, models
from app.service.file_service import FileService
from app.service.open_ai_service import OpenAiService
from flask import session
from langchain_text_splitters import RecursiveCharacterTextSplitter


class QdrantConnection:
    """
    Qdrant Connector
    """
    def __init__(self, url=None):
        """
        Initialize the Qdrant connection.
        :param url: The host address for the Qdrant server.
        """
        # Allow host and port to be specified or use defaults
        self.url = url or os.getenv('QDRANT_URL')
        self.client = self._connect()
        self.file_service = FileService()
        self.open_ai_service = OpenAiService()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def _connect(self) -> QdrantClient:
        """
        Establish a connection to the Qdrant server.

        :return: An instance of the QdrantClient.
        """
        return QdrantClient(url=self.url)

    def search(self, collection_name, query_vector, limit=3) -> list:
        """
        Perform a vector search in the specified Qdrant collection.

        :param collection_name: The name of the Qdrant collection to search.
        :param query_vector: The vector representation of the query.
        :param limit: The maximum number of results to return.
        :return: A list of payloads from the search results.
        """
        try:
            response = self.client.search(collection_name=collection_name, query_vector=query_vector, limit=limit)
            return [point.payload for point in response]
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

    def insert(self, collection_name, points):
        """
        Insert data points into the specified Qdrant collection.

        :param collection_name: The name of the Qdrant collection to insert data into.
        :param points: A list of data points to insert, each being a dictionary with `id`, `vector`, and `payload`.
        :return: None
        """
        try:
            self.client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            raise RuntimeError(f"Insertion failed: {e}") from e

    def delete(self, collection_name, point_ids):
        """
        Delete data points from the specified Qdrant collection.

        :param collection_name: The name of the Qdrant collection to delete data from.
        :param point_ids: A list of point IDs to delete.
        :return: None
        """
        try:
            self.client.delete(collection_name=collection_name, points_selector=point_ids)
        except Exception as e:
            raise RuntimeError(f"Deletion failed: {e}") from e

    def create_collection(self, collection_name="default", size=1536, distance=models.Distance.COSINE) -> None:
        """
        Creates new collection in Qdrant

        :param collection_name: What name should the new collection have
        :param size: Dimensions of vectors that the db should have, default 1536
        :param distance: default COSINE
        :return: None
        """
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=size, distance=distance))

    def get_collection_name(self, filename: str, suffix: str) -> str:
        """
        Returns concatenated collection name
        :param filename:
        :param suffix:
        :return:
        """
        return session['username'] + '_' + filename

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if specific collection exists
        :param collection_name:
        :return:
        """
        if self.client.collection_exists(collection_name):
            return True
        return False

    def collection_points_count(self, collection_name: str) -> int:
        """
        :param collection_name: To check specific collection
        :return: Number of points in collection if it exists
        """
        return self.client.count(collection_name).count if self.client.collection_exists(collection_name) else 0
    
    
    def run(self, collection_name: str, file_name: str):
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name)
        
        self.prepare_points(collection_name, file_name)
        
        
    def prepare_points(self, collection_name, file_name):
        collection_name = collection_name

        payload_id = 0

        data = self.file_service.load_pdf_content(filename=file_name) if file_name.endswith('.pdf') else self.file_service.load_txt_content(filename=file_name)


        if file_name.endswith('.pdf'):
            chunks = self.text_splitter.split_documents(data)
        else:
            chunks = self.text_splitter.split_documents([data])
            
        for chunk in chunks:
            payload_id += 1
            payload = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            vector = self.open_ai_service.get_embedding(chunk.page_content)
            point = [
                models.PointStruct(
                    id=payload_id,
                    payload=payload,
                    vector=vector
                )
            ]

            self.insert(collection_name=collection_name, points=point)

        