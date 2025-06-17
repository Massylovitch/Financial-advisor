from pathlib import Path

LLM_MODEL_ID = "tiiuae/falcon-7b-instruct"
LLM_QLORA_CHECKPOINT = "massyl/fin-falcon-7b-lora:1.0.0"
LLM_INFERNECE_MAX_NEW_TOKENS = 500
LLM_INFERENCE_TEMPERATURE = 1.0

VECTOR_DB_OUTPUT_COLLECTION_NAME = "alpaca_financial_news"

TEMPLATE_NAME = "falcon"

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MAX_INPUT_LENGTH = 384


CACHE_DIR = Path.home() / ".cache" / "tutorial"
