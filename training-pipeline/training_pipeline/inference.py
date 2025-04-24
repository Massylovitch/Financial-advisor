from pathlib import Path
from training_pipeline.prompter import get_llm_template
from tqdm import tqdm
from training_pipeline.utils import write_json
import logging
from training_pipeline.data_proc import FinanceDataset
from training_pipeline import models
import time, os
import opik


CACHE_DIR = Path.home() / ".cache" / "hands-on-llms"
logger = logging.getLogger(__name__)


class InferenceAPI:

    def __init__(
        self,
        peft_model_id,
        model_id,
        template_name,
        root_dataset_dir,
        test_dataset_file,
        name="inference-api",
        max_new_tokens=50,
        temperature=1.0,
        model_cache_dir=CACHE_DIR,
        debug=False,
        device="cuda:0",
    ):
        self._template_name = template_name
        self._prompt_template = get_llm_template(template_name)
        self._peft_model_id = peft_model_id
        self._model_id = model_id
        self._name = name
        self._root_dataset_dir = root_dataset_dir
        self._test_dataset_file = test_dataset_file
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._model_cache_dir = model_cache_dir
        self._debug = debug
        self._device = device

        self._model, self._tokenizer, self._peft_config = self.load_model()
        if self._root_dataset_dir is not None:
            self._dataset = self.load_data()
        else:
            self._dataset = None

        opik.configure(
            workspace=os.environ["COMET_WORKSPACE"],
            api_key=os.environ["COMET_API_KEY"],
        )

    @classmethod
    def from_config(cls, config, root_dataset_dir, model_cache_dir=CACHE_DIR):
        return cls(
            peft_model_id=config.peft_model["id"],
            model_id=config.model["id"],
            template_name=config.model["template_name"],
            root_dataset_dir=root_dataset_dir,
            test_dataset_file=config.dataset["file"],
            max_new_tokens=config.model["max_new_tokens"],
            temperature=config.model["temperature"],
            model_cache_dir=model_cache_dir,
            debug=config.setup.get("debug", False),
            device=config.setup.get("device", "cuda:0"),
        )

    def load_data(self):

        logger.info(f"Loading QA dataset from {self._root_dataset_dir=}")

        if self._debug:
            max_samples = 3
        else:
            max_samples = None

        dataset = FinanceDataset(
            data_path=self._root_dataset_dir / self._test_dataset_file,
            template=self._template_name,
            scope="inference",
            max_samples=max_samples,
        ).to_huggingface()

        logger.info(f"Loaded {len(dataset)} samples for inference")

        return dataset

    def load_model(self):
        logger.info(f"Loading model using {self._model_id=} and {self._peft_model_id=}")

        model, tokenizer, peft_config = models.build_qlora_model(
            pretrained_model_name_or_path=self._model_id,
            peft_pretrained_model_name_or_path=self._peft_model_id,
            gradient_checkpointing=False,
            cache_dir=self._model_cache_dir,
        )
        model.eval()

        return model, tokenizer, peft_config

    @opik.track
    def infer(self, infer_prompt, infer_payload):
        start_time = time.time()
        answer = models.prompt(
            model=self._model,
            tokenizer=self._tokenizer,
            input_text=infer_prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            device=self._device,
        )
        end_time = time.time()
        duration = end_time - start_time

        return answer

    @opik.track
    def infer_all(self, output_file=None):

        assert (
            self._dataset is not None
        ), "Dataset not loaded. Provide a dataset directory to the constructor: 'root_dataset_dir'."

        prompt_and_answers = []
        should_save_output = output_file is not None
        for sample in tqdm(self._dataset):
            answer = self.infer(
                infer_prompt=sample["prompt"], infer_payload=sample["payload"]
            )

            if should_save_output:
                prompt_and_answers.append(
                    {
                        "prompt": sample["prompt"],
                        "answer": answer,
                    }
                )

        if should_save_output:
            write_json(prompt_and_answers, output_file)
