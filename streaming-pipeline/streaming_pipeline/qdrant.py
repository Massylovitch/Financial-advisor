import os
from dotenv import load_dotenv
from typing import Optional

from bytewax.outputs import DynamicSink, StatelessSinkPartition
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from streaming_pipeline.models import Document
from qdrant_client.models import PointStruct
from streaming_pipeline import constants

load_dotenv()


class QdrantVectorOutput(DynamicSink):

    def __init__(
        self,
        vector_size=384,
        client=None,
        collection_name=constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):

        self._vector_size = vector_size
        self._collection_name = collection_name

        if client:
            self.client = QdrantClient(":memory:")
        else:
            self.client = build_qdrant_client()

        try:
            self.client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def build(self, step_id, worker_index, worker_count):
        return QdrantVectorSink(self.client)


class QdrantVectorSink(StatelessSinkPartition):

    def __init__(self, client):
        self._client = client
        self._collection_name = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME

    def write_batch(self, documents: Document):
        for document in documents:
            ids, payloads = document.to_payloads()
            points = [
                PointStruct(id=ids, vector=document.embeddings[0][0], payload=payloads)
            ]
            self._client.upsert(collection_name=self._collection_name, points=points)


def build_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
):

    try:
        url = os.environ["QDRANT_URL"]
    except KeyError:
        raise KeyError(
            "QDRANT_URL must be set as environment variable or manually passed as an argument."
        )

    try:
        api_key = os.environ["QDRANT_API_KEY"]
    except KeyError:
        raise KeyError(
            "QDRANT_API_KEY must be set as environment variable or manually passed as an argument."
        )

    client = QdrantClient(url, api_key=api_key)

    return client
