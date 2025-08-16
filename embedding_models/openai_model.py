from embedding_models.base import EmbeddingModel
from openai import OpenAI
import numpy as np
from typing import Union, List
from dotenv import load_dotenv
import os

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model_name
        self.client = self._init_client(api_key)
        self.embeddings: dict[str, np.ndarray] = {}

    def _init_client(self, api_key: str | None = None) -> OpenAI:
        if api_key:
            return OpenAI(api_key=api_key)
        load_dotenv()
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return OpenAI(api_key=env_key)
        # Fallback: let SDK resolve from environment/config
        return OpenAI()

    def embed(self, words: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(words, str):
            return self._get_embedding(words)
        return self._get_embeddings_batch(words)

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vectors: List[np.ndarray] = []
        # The API returns embeddings in the same order as inputs
        for i, item in enumerate(resp.data):
            vec = np.array(item.embedding, dtype=np.float32)
            word = texts[i]
            self.embeddings[word] = vec
            vectors.append(vec)
        return vectors

    def _get_embedding(self, word: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=word)
        vector = np.array(resp.data[0].embedding, dtype=np.float32)
        self.embeddings[word] = vector
        return vector

    def get_all_embeddings(self) -> dict:
        return self.embeddings