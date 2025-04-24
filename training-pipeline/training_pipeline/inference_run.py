from training_pipeline.inference import InferenceAPI
from pathlib import Path
import fire
from training_pipeline import configs
from beam import Volume, Image, function, Output

OUTPUT_PATH = "./output"


@function(
    name="inference_qa",
    volumes=[
        Volume(mount_path="./qa_dataset", name="qa_dataset"),
        Volume(mount_path="./model_cache", name="model_cache"),
        Volume(
            mount_path=OUTPUT_PATH,
            name="train_qa_output",
        ),
    ],
    secrets=["COMET_API_KEY", "COMET_WORKSPACE", "COMET_PROJECT_NAME"],
    image=Image(python_version="python3.10", python_packages="requirements.txt"),
    gpu="T4",
    cpu=4,
)
def infer(
    config_file,
    dataset_dir,
    output_dir="output_inference",
    model_cache_dir=None,
):

    config_file = Path(config_file)
    root_dataset_dir = Path(dataset_dir)
    output_dir = Path(f"{OUTPUT_PATH}/{output_dir}")
    model_cache_dir = Path(model_cache_dir) if model_cache_dir else None
    output_dir.mkdir(exist_ok=True, parents=True)

    inference_config = configs.InferenceConfig.from_yaml(config_file)
    inference_api = InferenceAPI.from_config(
        config=inference_config,
        root_dataset_dir=root_dataset_dir,
        model_cache_dir=model_cache_dir,
    )

    inference_api.infer_all(output_file=output_dir / "output-inference-api.json")
    output = Output(path=output_dir / "output-inference-api.json")
    output.save()
    output_url = output.public_url()
    print(output_url)


if __name__ == "__main__":
    fire.Fire(infer)
