from transformers import AutoModel, AutoTokenizer
from streaming_pipeline import constants


class EmbeddingModel:
    def __init__(
        self,
        model_name=constants.EMBEDDING_MODEL_ID,
        max_input_length=constants.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
        device=constants.EMBEDDING_MODEL_DEVICE,
        cache_dir=None,
    ):

        self._model_name = model_name
        self._max_input_name = max_input_length
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(
            self._device
        )
        self._model.eval()

    def __call__(self, input_text, to_list=True):

        tokenized_text = self._tokenizer(
            input_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._max_input_name,
        )  # .to_device(self._device)

        result = self._model(**tokenized_text)
        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()

        return embeddings
