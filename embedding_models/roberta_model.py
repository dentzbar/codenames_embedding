from embedding_models.base import EmbeddingModel
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
from typing import Union, List

class RobertaEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name="roberta-base"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.eval()
        self.embeddings = {}  # Will fill on demand

    def embed(self, words: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(words, str):
            return self._get_embedding(words)
        return [self._get_embedding(w) for w in words]

    def _get_embedding(self, word: str) -> np.ndarray:
        inputs = self.tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token (first vector) as sentence/word embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        self.embeddings[word] = cls_embedding
        return cls_embedding

    def get_all_embeddings(self) -> dict:
        return self.embeddings