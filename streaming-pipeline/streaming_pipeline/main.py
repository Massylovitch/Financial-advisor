import bytewax.operators as op
from bytewax.dataflow import Dataflow
from pydantic import TypeAdapter
from streaming_pipeline.qdrant import QdrantVectorOutput
from streaming_pipeline.models import NewsArticle
from streaming_pipeline.alpaca_batch import AlpacaNewsBatchInput
import datetime
from streaming_pipeline.embeddings import EmbeddingModel


def build_alpaca_news_flow(latest_n_days=1):

    to_datetime = datetime.datetime.now()
    from_datetime = to_datetime - datetime.timedelta(days=latest_n_days)

    model = EmbeddingModel()

    flow = Dataflow("financial_news_flow")
    stream = op.input("input", flow, AlpacaNewsBatchInput(from_datetime, to_datetime))
    articles = op.flat_map(
        "parse_message",
        stream,
        lambda messages: TypeAdapter(list[NewsArticle]).validate_python(messages),
    )
    documents = op.map(
        "convert to doc", articles, lambda article: article.to_document()
    )
    chunks = op.map("convert to chunks", documents, lambda doc: doc.compute_chunks())
    embeddings = op.map(
        "convert to embeddings", chunks, lambda chunk: chunk.compute_embeddings(model)
    )
    # op.inspect("help", embeddings)
    op.output("output", embeddings, QdrantVectorOutput())

    return flow
