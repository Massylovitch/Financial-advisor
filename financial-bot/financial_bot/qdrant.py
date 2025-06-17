import os
from typing import Optional

from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()


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
