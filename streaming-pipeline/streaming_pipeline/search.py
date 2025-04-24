from fire import Fire

from streaming_pipeline.qdrant import build_qdrant_client
from streaming_pipeline.embeddings import EmbeddingModel
from streaming_pipeline import constants


def search(query_string):

    client = build_qdrant_client()
    model = EmbeddingModel()

    query_embedding = model(query_string)[0]
    
    hits = client.query_points(
        collection_name=constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        query=query_embedding,
        limit=5,
    )

    for hit in hits.points:
       print(hit.payload["text"])


if __name__ == "__main__":
    Fire(search)