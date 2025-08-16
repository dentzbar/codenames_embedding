import numpy as np
from typing import Tuple, List

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def top_k_similar(
    target_embedding: np.ndarray,
    all_embeddings: dict,
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    Returns the top-k words most similar to the target embedding.
    all_embeddings: {word: vector}
    """
    similarities = []
    for word, vec in all_embeddings.items():
        sim = cosine_similarity(target_embedding, vec)
        similarities.append((word, sim))

    # Sort by similarity (descending) and return top k
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]