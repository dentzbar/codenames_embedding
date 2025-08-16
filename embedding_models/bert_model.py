from embedding_models.base import EmbeddingModel
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from typing import Union, List

class BertEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None, max_length: int = 12):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.max_length = max_length

    def embed(self, words: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(words, str):
            return self._get_embedding(words)
        # Batched embedding for a list of words
        return self._get_embeddings_batch(words)

    def _get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        # Use [CLS] token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
        return cls_emb

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        if len(texts) == 0:
            return []
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        cls_embs = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        # Return as list of np.ndarray per base interface
        return [cls_embs[i] for i in range(cls_embs.shape[0])] 