#!/usr/bin/env python3
"""
Combines words from simple_word_dataset.pkl with wordnet_words_n_v_a.pkl,
adding all simple words in lowercase to the wordnet collection.
"""

import os
import pickle
import sys
from typing import List

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils import load_words

def combine_word_datasets():
    # Load existing wordnet words
    wordnet_path = os.path.join("data", "wordnet_words_n_v_a.pkl")
    simple_path = os.path.join("data", "simple_word_dataset.pkl")
    
    if not os.path.exists(wordnet_path):
        print(f"WordNet file not found: {wordnet_path}")
        return
    
    if not os.path.exists(simple_path):
        print(f"Simple word dataset not found: {simple_path}")
        return
    
    # Load both datasets
    wordnet_words = load_words(wordnet_path)
    simple_words = load_words(simple_path)
    
    if wordnet_words is None or simple_words is None:
        print("Failed to load one or both word datasets")
        return
    
    print(f"Loaded {len(wordnet_words)} words from WordNet dataset")
    print(f"Loaded {len(simple_words)} words from simple dataset")
    
    # Convert to sets for efficient operations, ensure lowercase
    wordnet_set = set(w.lower() for w in wordnet_words)
    simple_set = set(w.lower() for w in simple_words)
    
    # Combine the sets
    combined_set = wordnet_set.union(simple_set)
    combined_words = sorted(list(combined_set))
    
    print(f"Combined dataset contains {len(combined_words)} unique words")
    print(f"Added {len(combined_set) - len(wordnet_set)} new words from simple dataset")
    
    # Save the combined dataset back to the wordnet file
    with open(wordnet_path, "wb") as f:
        pickle.dump(combined_words, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Updated {wordnet_path} with combined word list")




def add_simple_embeddings():
    """Add embeddings for simple words to the existing RoBERTa vector store."""
    import pickle
    from embedding_models.roberta_model import RobertaEmbeddingModel
    import numpy as np
    from tqdm import tqdm
    
    # Paths
    simple_path = os.path.join("data", "simple_word_dataset.pkl")
    vectorstore_path = os.path.join("data", "wordnet_roberta_embedded_store.pkl")
    
    if not os.path.exists(simple_path):
        print(f"Simple word dataset not found: {simple_path}")
        return
    
    if not os.path.exists(vectorstore_path):
        print(f"RoBERTa vector store not found: {vectorstore_path}")
        return
    
    # Load simple words
    simple_words = load_words(simple_path)
    if simple_words is None:
        print("Failed to load simple word dataset")
        return
    
    # Load existing vector store
    with open(vectorstore_path, "rb") as f:
        existing_vectors = pickle.load(f)
    
    print(f"Loaded existing vector store with {len(existing_vectors)} words")
    print(f"Loaded {len(simple_words)} words from simple dataset")
    
    # Find words that need embedding (not already in vector store)
    simple_words_lower = [w.lower() for w in simple_words]
    words_to_embed = [w for w in simple_words_lower if w not in existing_vectors]
    
    if not words_to_embed:
        print("All simple words already have embeddings in the vector store")
        return
    
    print(f"Need to embed {len(words_to_embed)} new words")
    
    # Initialize RoBERTa model
    model = RobertaEmbeddingModel(model_name="roberta-base")
    
    # Embed new words in batches
    batch_size = 64
    new_vectors = {}
    
    for i in tqdm(range(0, len(words_to_embed), batch_size), desc="Embedding batches"):
        batch = words_to_embed[i:i + batch_size]
        try:
            embeddings = model.embed(batch)
            for word, emb in zip(batch, embeddings):
                if emb is not None:
                    new_vectors[word] = np.asarray(emb, dtype=np.float32)
        except Exception as e:
            print(f"Error embedding batch starting at index {i}: {e}")
            # Try individual words in this batch
            for word in tqdm(batch, desc="Retrying individual words", leave=False):
                try:
                    emb = model.embed(word)
                    if emb is not None:
                        new_vectors[word] = np.asarray(emb, dtype=np.float32)
                except Exception as e2:
                    print(f"  Skipping word '{word}': {e2}")
    
    # Combine with existing vectors
    combined_vectors = {**existing_vectors, **new_vectors}
    
    # Save updated vector store
    with open(vectorstore_path, "wb") as f:
        pickle.dump(combined_vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Added {len(new_vectors)} new embeddings")
    print(f"Updated vector store now contains {len(combined_vectors)} words")
    print(f"Saved to {vectorstore_path}")


 # Check if words are in the wordnet vocabulary
def check_words_in_wordnet(words_to_check: List[str], wordnet_path: str = "data/wordnet_words_n_v_a.pkl") -> None:
    """Check if a list of words exists in the WordNet vocabulary."""
    try:
        wordnet_words = load_words(wordnet_path)
        if wordnet_words is None:
            print(f"Failed to load WordNet vocabulary from {wordnet_path}")
            return
        
        wordnet_words_lower = [w.lower() for w in wordnet_words]
        wordnet_set = set(wordnet_words_lower)
        
        words_to_check_lower = [w.lower() for w in words_to_check]
        
        found_words = [w for w in words_to_check_lower if w in wordnet_set]
        missing_words = [w for w in words_to_check_lower if w not in wordnet_set]
        
        print(f"\nChecking {len(words_to_check)} words against WordNet vocabulary:")
        print(f"WordNet vocabulary size: {len(wordnet_words)}")
        print(f"Found: {len(found_words)} words")
        print(f"Missing: {len(missing_words)} words")
        
        if missing_words:
            print(f"\nMissing words: {missing_words[:20]}")  # Show first 20
            if len(missing_words) > 20:
                print(f"... and {len(missing_words) - 20} more")
        
    except Exception as e:
        print(f"Error checking words in WordNet: {e}")

if __name__ == "__main__":
      
    # combine_word_datasets()
    # add_simple_embeddings()
   
    # Example usage - check some test words
    test_words = ["revealed"]
    check_words_in_wordnet(test_words)
