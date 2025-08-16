from embedding_models.base import EmbeddingModel
from typing import Union, List, Optional
import numpy as np
from dotenv import load_dotenv
import os

# Google Gemini Python SDK
# pip install google-genai
from google import genai


class GeminiEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",  # or "RETRIEVAL_QUERY"
        output_dimensionality: Optional[int] = 1536,  # default 1536 dimensions, options: 768/1536/3072
    ) -> None:
        self.model_name = model_name
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        self.client = self._init_client(api_key)
        self.embeddings: dict[str, np.ndarray] = {}

    def _init_client(self, api_key: Optional[str]) -> genai.Client:
        if api_key:
            return genai.Client(api_key=api_key)
        load_dotenv()
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return genai.Client(api_key=env_key)
        # Let SDK try other mechanisms (though Gemini typically needs explicit key)
        return genai.Client()

    def embed(self, words: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(words, str):
            return self._get_embedding(words)
        return self._get_embeddings_batch(words)

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        config = {"task_type": self.task_type}
        if self.output_dimensionality is not None:
            config["output_dimensionality"] = int(self.output_dimensionality)

        res = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=config,
        )
        # res.embeddings is a list of objects with .values
        vectors: List[np.ndarray] = []
        for i, emb in enumerate(res.embeddings):
            vec = np.asarray(emb.values, dtype=np.float32)
            word = texts[i]
            self.embeddings[word] = vec
            vectors.append(vec)
        return vectors

    def _get_embedding(self, text: str) -> np.ndarray:
        config = {"task_type": self.task_type}
        if self.output_dimensionality is not None:
            config["output_dimensionality"] = int(self.output_dimensionality)
        res = self.client.models.embed_content(
            model=self.model_name,
            contents=[text],
            config=config,
        )
        vec = np.asarray(res.embeddings[0].values, dtype=np.float32)
        self.embeddings[text] = vec
        return vec

    def get_all_embeddings(self) -> dict:
        return self.embeddings 