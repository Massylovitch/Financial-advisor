import bytewax.operators as op
import sample
from bytewax.connectors.stdio import StdOutSink
from bytewax.dataflow import Dataflow
from bytewax.testing import TestingSource
from pydantic import TypeAdapter

from streaming_pipeline.models import NewsArticle


flow = Dataflow("financial_news_flow")
stream = op.input("input", flow, TestingSource(sample.financial_news))
articles = op.flat_map(
    "parse_message",
    stream,
    lambda messages: TypeAdapter(list[NewsArticle]).validate_python(messages),
)
documents = op.map("convert to doc", articles, lambda article: article.to_document())
chunks = op.map("convert to chunks", documents, lambda article: article.compute_chunks())
chunks = op.map("convert to embeddings", documents, lambda article: article.compute_embeddings())
op.inspect("help", documents)
# op.output("output", stream, StdOutSink())
