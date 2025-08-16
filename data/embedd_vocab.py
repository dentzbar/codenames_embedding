#!/usr/bin/env python3
"""
Embeds a vocabulary list into vectors using a selectable embedding model
and saves the vector store as a pickle file.

Default input: data/wordnet_words_n_v_a.pkl
Default model: OpenAI (text-embedding-3-small)
Default output: data/wordnet_<model>_embedded_store.pkl

Usage (CLI mode):
    python data/embedd_vocab.py --model openai --model_name text-embedding-3-small \
        --input data/wordnet_words_n_v_a.pkl --output data/wordnet_openai_embedded_store.pkl --batch_size 64

Models:
    - bert (bert-base-uncased)
    - roberta (roberta-base)
    - openai (text-embedding-3-small)
    - gemini (gemini-embedding-001)

Notes:
    - For large vocabularies (~75k+), this may take time and memory.
    - Use batch processing and periodic checkpointing to avoid data loss.
    - OpenAI model requires OPENAI_API_KEY in environment (or .env loaded elsewhere).
    - Gemini model requires GEMINI_API_KEY in environment.
"""

import os
import sys
import argparse
import pickle
import time
from tqdm import tqdm
from typing import List, Dict

import numpy as np

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils import load_words
from embedding_models.bert_model import BertEmbeddingModel
from embedding_models.roberta_model import RobertaEmbeddingModel
from embedding_models.openai_model import OpenAIEmbeddingModel
from embedding_models.gemini_model import GeminiEmbeddingModel

MODEL_CHOICES = {
    "bert": (BertEmbeddingModel, "bert-base-uncased"),
    "roberta": (RobertaEmbeddingModel, "roberta-base"),
    "openai": (OpenAIEmbeddingModel, "text-embedding-3-small"),
    "gemini": (GeminiEmbeddingModel, "gemini-embedding-001"),
}

def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def embed_vocab(
    model_group: str,
    model_name: str,
    input_path: str,
    output_path: str,
    batch_size: int = 64,
    checkpoint_every: int = 2000,
) -> None:
    # Load words
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    words = load_words(input_path)
    if words is None or len(words) == 0:
        print("No words loaded from input file")
        sys.exit(1)

    print(f"Loaded {len(words)} words from {input_path}")

    # Initialize model
    if model_group not in MODEL_CHOICES:
        raise ValueError(f"Unknown model group '{model_group}'. Choose from: {list(MODEL_CHOICES.keys())}")
    model_cls, default_name = MODEL_CHOICES[model_group]
    effective_model_name = model_name or default_name
    print(f"Initializing model: {model_group} ({effective_model_name})")
    # All wrappers accept model_name
    model = model_cls(model_name=effective_model_name)  # type: ignore[arg-type]

    # Prepare storage
    vectors: Dict[str, np.ndarray] = {}
    start_time = time.time()
    processed = 0

    # Process in batches
    total_batches = (len(words) + batch_size - 1) // batch_size
    for batch_idx, batch in enumerate(tqdm(chunk_list(words, batch_size), total=total_batches, desc="Embedding words")):
        try:
            embeddings = model.embed(batch)
        except Exception as e:
            print(f"Error embedding batch {batch_idx}: {e}")
            # Try individual embedding to skip bad tokens
            embeddings = []
            for w in tqdm(batch, desc="Retrying individual words", leave=False):
                try:
                    emb = model.embed(w)
                    embeddings.append(emb)
                except Exception as e2:
                    print(f"  Skipping word '{w}': {e2}")
                    embeddings.append(None)

        for w, emb in zip(batch, embeddings):
            if emb is None:
                continue
            vectors[w] = np.asarray(emb, dtype=np.float32)
            processed += 1

        # Checkpoint
        if processed % checkpoint_every == 0:
            tmp_path = output_path + ".checkpoint.pkl"
            with open(tmp_path, "wb") as f:
                pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed = time.time() - start_time
            print(f"Checkpoint: {processed} words embedded, elapsed {elapsed/60:.1f} min, saved to {tmp_path}")

    # Save final store
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - start_time
    print(f"Saved vector store with {len(vectors)} words to {output_path}")
    print(f"Total time: {elapsed/60:.1f} minutes")

def main():
    parser = argparse.ArgumentParser(description="Embed vocabulary and save vector store")
    parser.add_argument("--model", type=str, default="openai", choices=list(MODEL_CHOICES.keys()), help="Embedding model group to use")
    parser.add_argument("--model_name", type=str, default=None, help="Specific model name, e.g. 'text-embedding-3-small' for OpenAI or 'gemini-embedding-001' for Gemini")
    parser.add_argument("--input", type=str, default=os.path.join("data", "wordnet_words_n_v_a.pkl"), help="Path to input word list pickle")
    parser.add_argument("--output", type=str, default=None, help="Path to output vector store pickle; defaults to data/wordnet_<model>_embedded_store.pkl")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding")
    parser.add_argument("--checkpoint_every", type=int, default=2000, help="Checkpoint every N words")
    args = parser.parse_args()

    # Derive default output if not provided
    output_path = args.output or os.path.join("data", f"wordnet_{args.model}_embedded_store.pkl")

    embed_vocab(
        model_group=args.model,
        model_name=args.model_name or MODEL_CHOICES[args.model][1],
        input_path=args.input,
        output_path=output_path,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )

if __name__ == "__main__":
    # If run without CLI args (len(sys.argv) == 1), use these inline defaults;
    # Otherwise, argparse in main() will handle inputs
    if len(sys.argv) == 1:
        chosen_model_group = "gemini"             # "bert" | "roberta" | "openai" | "gemini"
        chosen_model_name = "gemini-embedding-001"
        input_path = os.path.join("data", "wordnet_words_n_v_a.pkl")
        output_path = os.path.join("data", f"wordnet_{chosen_model_group}_embedded_store.pkl")
        batch_size = 64
        checkpoint_every = 2000
        embed_vocab(
            model_group=chosen_model_group,
            model_name=chosen_model_name,
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
        )
    else:
        main() 
    