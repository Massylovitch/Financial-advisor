from comet_ml import API
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os
from peft import LoraConfig, PeftModel
from pathlib import Path


CACHE_DIR = Path.home() / ".cache" / "tutorial"


def build_qlora_model(
    pretrained_model_name_or_path="tiiuae/falcon-7b-instruct",
    peft_pretrained_model_name_or_path=None,
    gradient_checkpointing=True,
    cache_dir=None,
):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        revision="main",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=False,
        truncation=True,
        cache_dir=str(cache_dir) if cache_dir else None,
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

        lora_config = LoraConfig.from_pretrained(peft_pretrained_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_pretrained_model_name_or_path)
    else:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["query_key_value"],
        )

    return model, tokenizer, lora_config


def download_from_model_registry(model_id, cache_dir=None):

    if cache_dir is None:
        cache_dir = CACHE_DIR
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


def prompt(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    device: str = "cuda:0",
):
    inputs = tokenizer(input_text, return_tensors="pt", return_token_type_ids=False).to(
        device
    )
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=temperature
    )

    output = outputs[0]
    input_ids = inputs.input_ids
    input_length = input_ids.shape[-1]
    output = output[input_length:]
    output = tokenizer.decode(output, skip_special_tokens=True)
    return output
