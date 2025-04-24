from training_pipeline.utils import load_json
from dataclasses import asdict, dataclass, field
from training_pipeline.prompter import get_llm_template
from datasets import Dataset


@dataclass(frozen=True)
class DataSample:
    user_context: str = field(repr=False)
    news_context: str = ""
    chat_history: str = ""
    question: str = ""
    answer: str = ""


class FinanceDataset:
    def __init__(
        self,
        data_path,
        scope,
        template="falcon",
        max_samples=None,
    ):

        self._data_path = data_path
        self._scope = scope
        self._max_samples = max_samples
        self._template = get_llm_template(template)
        self._raw_data = self.load(data_path)

    def load(self, data_path):

        data = load_json(data_path)
        if self._max_samples is not None:
            data = data[: self._max_samples]

        return self.deserialize(data)

    def deserialize(self, data):

        if self._scope == "training":
            return [
                DataSample(
                    user_context=sample["about_me"],
                    news_context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                    answer=sample["response"],
                )
                for sample in data
            ]
        else:
            return [
                DataSample(
                    user_context=sample["about_me"],
                    news_context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                )
                for sample in data
            ]

    def to_huggingface(self):

        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        if self._scope == "training":
            template_mapping_func = self._template.format_train
        else:
            template_mapping_func = self._template.format_infer

        dataset = dataset.map(
            template_mapping_func, remove_columns=dataset.column_names
        )

        return dataset


if __name__ == "__main__":
    from pathlib import Path

    training_dataset = FinanceDataset(
        data_path=Path("datasets") / "training_data.json",
        template="falcon",
        scope="training",
    ).to_huggingface()

    print(training_dataset["prompt"][0])
    print("------------------------------------------------------")
    print(training_dataset["payload"][0])
