from dataclasses import dataclass
from typing import Any, Dict
from training_pipeline.utils import load_yaml
from trl import SFTConfig

@dataclass
class TrainingConfig:
    training: SFTConfig
    model: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path, output_dir):

        config = load_yaml(config_path)

        config["training"] = cls._dict_to_training_arguments(
            training_config=config["training"], output_dir=output_dir
        )

        return cls(**config)

    @classmethod
    def _dict_to_training_arguments(cls, training_config, output_dir):

        return SFTConfig(
            output_dir=str(output_dir),
            logging_dir=str(output_dir / "logs"),
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            eval_accumulation_steps=training_config["eval_accumulation_steps"],
            optim=training_config["optim"],
            save_steps=training_config["save_steps"],
            logging_steps=training_config["logging_steps"],
            learning_rate=training_config["learning_rate"],
            fp16=training_config["fp16"],
            max_grad_norm=training_config["max_grad_norm"],
            num_train_epochs=training_config["num_train_epochs"],
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            eval_strategy=training_config["evaluation_strategy"],
            eval_steps=training_config["eval_steps"],
            report_to=training_config["report_to"],
            seed=training_config["seed"],
            load_best_model_at_end=training_config["load_best_model_at_end"],
            dataset_text_field=training_config["dataset_text_field"],
            packing=training_config["packing"],
            eos_token=training_config["eos_token"],
            max_seq_length=training_config["max_seq_length"],
        )


@dataclass
class InferenceConfig:

    model: Dict[str, Any]
    peft_model: Dict[str, Any]
    setup: Dict[str, Any]
    dataset: Dict[str, str]

    @classmethod
    def from_yaml(cls, config_path):

        config = load_yaml(config_path)

        return cls(**config)
