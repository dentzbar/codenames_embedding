# codenames_embeddings/embedding_models/base.py
from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np

class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, words: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Given a word or list of words, return the corresponding embedding(s).
        """
        pass