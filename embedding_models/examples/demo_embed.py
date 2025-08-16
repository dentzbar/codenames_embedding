import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embedding_models.engine import EmbeddingEngine
from embedding_models.glove_model import GloveEmbeddingModel
from embedding_models.roberta_model import RobertaEmbeddingModel
from embedding_models.openai_model import OpenAIEmbeddingModel
from utils.similarity import top_k_similar

def test_model(name, model):
    print(f"\n==== {name} ====")
    engine = EmbeddingEngine(model)
    words = ["king", "queen", "apple"]
    vecs = engine.get_embedding(words)

    for word, vec in zip(words, vecs):
        print(f"{word}: {vec[:5]}...")

    target_vec = engine.get_embedding("king")
    all_vecs = engine.model.get_all_embeddings()

    print(f"\nTop 5 similar to 'king' using {name}:")
    for w, s in top_k_similar(target_vec, all_vecs, k=5):
        print(f"{w}: {s:.4f}")

if __name__ == "__main__":
    # GloVe
    glove_path = os.path.join(os.path.dirname(__file__), "..", "data", "glove", "glove.6B.300d.txt")
    glove_path = os.path.abspath(glove_path)
    test_model("GloVe", GloveEmbeddingModel(glove_path))

    # RoBERTa
    test_model("RoBERTa", RobertaEmbeddingModel())

    # OpenAI
    test_model("OpenAI", OpenAIEmbeddingModel())