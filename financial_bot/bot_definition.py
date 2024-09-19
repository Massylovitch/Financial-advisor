from chains import ContextExtractorChain, FinancialBotQAChain, StatelessMemorySequentialChain
from embeddings import EmbeddingModel 
from qdrant import build_qdrant_client
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from template import get_llm_template
from langchain.memory import ConversationBufferWindowMemory
from typing import Callable, Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


llm_model_id = "tiiuae/falcon-7b-instruct"
vector_collection_name = "alpaca_financial_news"
template_name = "falcon"
debug = True

def build_huggingface_pipeline(
    temperature,
    max_new_tokens,
    debug=True
):
    if debug is True:
        return HuggingFacePipeline(
                pipeline=MockedPipeline(f=lambda _: "You are doing great!")
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            revision="main",
            device_map="auto",
            trust_remote_code=False,
            offload_folder="offload"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            llm_model_id,
            trust_remote_code=False,
            truncation=True,
        )
            
        model.eval()

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        hf = HuggingFacePipeline(pipeline=pipe)
        return hf

class MockedPipeline:

    task: str = "text-generation"

    def __init__(self, f: Callable[[str], str]):
        self.f = f

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        """
        Calls the pipeline with a given prompt and returns a list of generated text.

        Parameters:
        -----------
        prompt : str
            The prompt string to generate text from.

        Returns:
        --------
        List[Dict[str, str]]
            A list of dictionaries, where each dictionary contains a generated_text key with the generated text string.
        """

        result = self.f(prompt)

        return [{"generated_text": f"{prompt}{result}"}]

class FinancialBot:

    def __init__(self,):
        self._llm_model_id = llm_model_id
        self._embedding_model = EmbeddingModel()
        self._qdrant_client = build_qdrant_client()
        self._vector_collection_name = vector_collection_name
        self._llm_template_name = template_name

        self._llm_agent = build_huggingface_pipeline(
            max_new_tokens=500,
            temperature=1.0,
            debug=debug
        )
        self._llm_template = get_llm_template(name=self._llm_template_name)
        self.finbot_chain = self.build_chain()


    def build_chain(self):

        context_retrieval_chain = ContextExtractorChain(
            embedding_model=self._embedding_model,
            vector_store=self._qdrant_client,
            vector_collection=self._vector_collection_name,
            top_k=1,
        )
         
        llm_generator_chain = FinancialBotQAChain(
            hf_pipeline=self._llm_agent,
            template=self._llm_template,
        )
         
        seq_chain = StatelessMemorySequentialChain(
            memory=ConversationBufferWindowMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                k=3,
            ),
            chains=[context_retrieval_chain, llm_generator_chain],
            input_variables=["about_me", "question", "to_load_history"],
            output_variables=["answer"],
            verbose=True,
        )

        return seq_chain
    
    def answer(
        self,
        about_me,
        question,
        to_load_history = None
    ):
        inputs = {
            "about_me": about_me,
            "question": question,
            "to_load_history": to_load_history if to_load_history else [],
        }
        response = self.finbot_chain.run(inputs)

        return response