from financial_bot.chains import (
    ContextExtractorChain,
    FinancialBotQAChain,
    StatelessMemorySequentialChain,
)
from financial_bot.embeddings import EmbeddingModel
from financial_bot.qdrant import build_qdrant_client
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from financial_bot.template import get_llm_template
from langchain.memory import ConversationBufferWindowMemory
from typing import Callable, Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from financial_bot import constants
import torch
import os
from comet_ml import API
from pathlib import Path
from peft import LoraConfig, PeftConfig, PeftModel
from financial_bot import constants


def build_huggingface_pipeline(
    llm_model_id=constants.LLM_MODEL_ID,
    peft_pretrained_model_name_or_path=constants.LLM_QLORA_CHECKPOINT,
    temperature=constants.LLM_INFERENCE_TEMPERATURE,
    max_new_tokens=constants.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
    cache_dir=None,
    debug=True,
):

    if debug is True:
        return HuggingFacePipeline(
            pipeline=MockedPipeline(f=lambda _: "You are doing great!")
        )
    else:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            revision="main",
            device_map="auto",
            trust_remote_code=False,
            offload_folder="offload",
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            llm_model_id,
            trust_remote_code=False,
            truncation=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            with torch.no_grad():
                model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id

        if peft_pretrained_model_name_or_path:
            is_model_name = not os.path.isdir(peft_pretrained_model_name_or_path)
            if is_model_name:
                peft_pretrained_model_name_or_path = download_from_model_registry(
                    model_id=peft_pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                )
            print(
                "peft_pretrained_model_name_or_path", peft_pretrained_model_name_or_path
            )
            model = PeftModel.from_pretrained(model, peft_pretrained_model_name_or_path)

        model.gradient_checkpointing_disable()
        model.config.use_cache = True
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


def download_from_model_registry(model_id, cache_dir=None):

    if cache_dir is None:
        cache_dir = constants.CACHE_DIR
    output_folder = cache_dir / "models" / model_id

    already_downloaded = output_folder.exists()
    if not already_downloaded:
        workspace, model_id = model_id.split("/")
        model_name, version = model_id.split(":")

        api = API()
        model = api.get_model(workspace=workspace, model_name=model_name)
        model.download(version=version, output_folder=output_folder, expand=True)

    subdirs = [d for d in output_folder.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        model_dir = subdirs[0]
    else:
        raise RuntimeError(
            f"There should be only one directory inside the model folder. \
                Check the downloaded model at: {output_folder}"
        )

    return model_dir


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

    def __init__(self, debug=False):

        self._llm_model_id = constants.LLM_MODEL_ID
        self._vector_collection_name = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME
        self._llm_template_name = constants.TEMPLATE_NAME
        self._llm_inference_max_new_tokens = constants.LLM_INFERNECE_MAX_NEW_TOKENS
        self._llm_inference_temperature = constants.LLM_INFERENCE_TEMPERATURE

        self._embedding_model = EmbeddingModel()

        self._qdrant_client = build_qdrant_client()

        self._llm_agent = build_huggingface_pipeline(
            max_new_tokens=self._llm_inference_max_new_tokens,
            temperature=self._llm_inference_temperature,
            cache_dir=None,
            debug=debug,
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

    def answer(self, about_me, question, to_load_history=None):
        inputs = {
            "about_me": about_me,
            "question": question,
            "to_load_history": to_load_history if to_load_history else [],
        }
        response = self.finbot_chain.run(inputs)

        return response
