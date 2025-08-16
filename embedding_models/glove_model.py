import numpy as np
from typing import Union, List
from embedding_models.base import EmbeddingModel

class GloveEmbeddingModel(EmbeddingModel):
    def __init__(self, glove_path: str = "data/glove/glove.6B.300d.txt"):
        self.glove_path = glove_path
        self.embeddings = {}
        self.dim = 300
        self._load_glove()

    def _load_glove(self):
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                self.embeddings[word] = vector
        print(f"[GloVe] Loaded {len(self.embeddings)} words.")

    def embed(self, words: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(words, str):
            return self._get_embedding(words)
        return [self._get_embedding(word) for word in words]

    def _get_embedding(self, word: str) -> np.ndarray:
        vector = self.embeddings.get(word.lower())
        if vector is None:
            raise ValueError(f"[GloVe] Word '{word}' not found in GloVe vocabulary.")
        return vector
    
    def get_all_embeddings(self) -> dict:
        return self.embeddings