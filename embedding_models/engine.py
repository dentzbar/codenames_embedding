from typing import Union, List
import numpy as np
from embedding_models.base import EmbeddingModel

class EmbeddingEngine:
    def __init__(self, model: EmbeddingModel):
        self.model = model

    def get_embedding(self, words: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        return self.model.embed(words)