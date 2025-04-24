from training_pipeline.training import TrainingAPI
from pathlib import Path
import fire
from training_pipeline import configs
from beam import Volume, Image, function


@function(
    name="train_qa",
    volumes=[
        Volume(mount_path="./qa_dataset", name="qa_dataset"),
        Volume(
            mount_path="./output",
            name="train_qa_output",
        ),
        Volume(mount_path="./model_cache", name="model_cache"),
    ],
    secrets=["COMET_API_KEY", "COMET_WORKSPACE", "COMET_PROJECT_NAME"],
    image=Image(python_version="python3.12", python_packages="requirements.txt"),
    gpu="T4",
    cpu=4,
)
def train(
    config_file: str,
    output_dir: str,
    dataset_dir: str,
    model_cache_dir: str = None,
):
    config_file = Path(config_file)
    output_dir = Path(output_dir)
    root_dataset_dir = Path(dataset_dir)
    model_cache_dir = Path(model_cache_dir) if model_cache_dir else None

    training_config = configs.TrainingConfig.from_yaml(config_file, output_dir)
    training_api = TrainingAPI.from_config(
        config=training_config,
        root_dataset_dir=root_dataset_dir,
        model_cache_dir=model_cache_dir,
    )

    training_api.train()


if __name__ == "__main__":
    fire.Fire(train)
