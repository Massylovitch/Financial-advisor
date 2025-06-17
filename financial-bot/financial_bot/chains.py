from langchain.chains.base import Chain
from langchain.chains.sequential import SequentialChain
import qdrant_client
from financial_bot.embeddings import EmbeddingModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from financial_bot.template import PromptTemplate


class ContextExtractorChain(Chain):

    top_k: int = 1
    vector_store: qdrant_client.QdrantClient
    vector_collection: str
    embedding_model: EmbeddingModel

    @property
    def input_keys(self):
        return ["about_me", "question"]

    @property
    def output_keys(self):
        return ["context"]

    def _call(self, inputs):

        _, question_key = self.input_keys
        question = inputs[question_key]

        question = question[: self.embedding_model._max_input_length]
        embeddings = self.embedding_model(question)

        points_match = self.vector_store.query_points(
            query=embeddings,
            limit=self.top_k,
            collection_name=self.vector_collection,
        )

        context = ""
        for point in points_match.points:
            context += point.payload["text"] + "\n"

        return {
            "context": context,
        }


class FinancialBotQAChain(Chain):

    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate

    @property
    def input_keys(self):
        return ["context"]

    @property
    def output_keys(self):
        return ["answer"]

    def _call(self, inputs):
        prompt = self.template.format_infer(
            {
                "user_context": inputs["about_me"],
                "news_context": inputs["context"],
                "chat_history": inputs["chat_history"],
                "question": inputs["question"],
            }
        )

        response = self.hf_pipeline(prompt["prompt"])

        return {"answer": response}


class StatelessMemorySequentialChain(SequentialChain):

    def _call(self, inputs, **kwargs):
        history_input_keys = "to_load_history"
        to_load_history = inputs[history_input_keys]

        for human, ai in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )

        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[history_input_keys]

        return super()._call(inputs, **kwargs)

    def prep_outputs(self, inputs, outputs, return_only_outputs=False):

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results
