from utils import load_words
import random
import pickle
import numpy as np
from itertools import combinations
import time



#########################
### @@ SETUP @@ ###
#########################

# How many agent words to target with the code word
num_agent_words = 3

#########################################
### @@ Load full vocab_vectorestore@@ ###
#########################################

# Load the pre-computed BERT embeddings
with open('data/wordnet_bert_embedded_store.pkl', 'rb') as f:
    vocab_embeddings = pickle.load(f)

# Helper: L2 normalize vectors
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# Build a normalized embedding dict once for cosine as dot
normalized_vocab = {w: l2_normalize(v) for w, v in vocab_embeddings.items()}


############################
### @@ Load game words@@ ###
############################

# Load the word dataset
game_words = load_words('data/simple_word_dataset.pkl')

# Sample words for each category
agents = [w.lower() for w in random.sample(game_words, 9)]
inocents = [w.lower() for w in random.sample([w for w in game_words if w not in agents], 16)]
# Print agents with first num_agent_words in bold blue, rest in regular blue
agents_display = []
for i, agent in enumerate(agents):
    if i < num_agent_words:
        agents_display.append(f"\033[1;34m{agent}\033[0m")  # bold blue
    else:
        agents_display.append(f"\033[34m{agent}\033[0m")     # regular blue

print(f"Agents: {', '.join(agents_display)}")
print(f"Inocents: {', '.join(inocents)}")

#########################
### @@ ENCODER RUN @@ ###
#########################

# Prepare lowercase words and ensure presence in vocab
agent_candidate_pool = [w.lower() for w in agents]
innocent_words = [w.lower() for w in inocents]

# Filter out words not present in vocab
valid_agent_pool = [w for w in agent_candidate_pool if w in normalized_vocab]
innocent_words = [w for w in innocent_words if w in normalized_vocab]

if len(valid_agent_pool) < num_agent_words:
    raise ValueError("Not enough selected agent words exist in the embedding store for the chosen combination size. Try again.")

print(f"\n\033[1;35m===== 🎯 ENCODER =====\033[0m")

# Innocent matrix for fast max similarity computation (shared across combos)
if len(innocent_words) == 0:
    raise ValueError("No innocent words found in the embedding store.")
innoc_mat = np.stack([normalized_vocab[w] for w in innocent_words], axis=0)  # shape (M, D)

print(f"Vocab size: {len(normalized_vocab)}, Innocent words: {len(innocent_words)}")
print(f"Agent candidates: {len(valid_agent_pool)}, Combinations to evaluate: {len(list(combinations(valid_agent_pool, num_agent_words)))}")

start_time = time.time()

# OPTIMIZED VERSION: Precompute all matrices and use vectorized operations
# Build full candidate pool (excluding board words)
board_words = set(valid_agent_pool + innocent_words)
candidate_words = [w for w in normalized_vocab.keys() if w not in board_words]
print(f"Candidate words: {len(candidate_words)}")

if len(candidate_words) == 0:
    raise ValueError("No candidate words available after excluding board words.")

# Precompute candidate matrix (N x D)
V = np.stack([normalized_vocab[w] for w in candidate_words], axis=0)

# Precompute agent pool matrix (A x D) where A = number of agent candidates
agent_pool_mat = np.stack([normalized_vocab[w] for w in valid_agent_pool], axis=0)

# Precompute all pairwise similarities between candidates and agents (N x A)
cand_agent_sims = V @ agent_pool_mat.T

# Precompute all candidate-innocent similarities and take max (N,)
cand_innoc_sims = V @ innoc_mat.T  # (N x M)
neg_max_global = np.max(cand_innoc_sims, axis=1)  # (N,) - max innocent similarity per candidate

# Now evaluate combinations efficiently
best_global_margin = -1e9
best_combo = None
best_word = None
best_vec = None
best_candidate_words = candidate_words
best_margins_arr = None
best_pos_min_arr = None
best_neg_max_arr = neg_max_global

combo_count = 0
for combo in combinations(range(len(valid_agent_pool)), num_agent_words):
    combo_count += 1
    
    # Get similarities for this combination's agents (N x K)
    combo_sims = cand_agent_sims[:, combo]
    
    # Minimum similarity across the selected agents (N,)
    pos_min = np.min(combo_sims, axis=1)
    
    # Margin = min positive similarity - max innocent similarity
    margins = pos_min - neg_max_global
    
    # Best candidate for this combo
    local_best_idx = int(np.argmax(margins))
    local_best_margin = float(margins[local_best_idx])
    
    # Update global best if improved
    if local_best_margin > best_global_margin:
        best_global_margin = local_best_margin
        best_combo = [valid_agent_pool[i] for i in combo]
        best_word = candidate_words[local_best_idx]
        best_vec = normalized_vocab[best_word]
        best_margins_arr = margins
        best_pos_min_arr = pos_min

optimization_time = time.time() - start_time
print(f"Optimization completed in {optimization_time:.2f} seconds ({combo_count} combinations)")

# Validate we found a best combo/word
if best_combo is None or best_word is None:
    raise ValueError("Failed to find a valid code word for any agent word combination.")

# Selected target (agent) words based on the best combination
agent_words = best_combo
agent_words_display = [f"\033[1;34m{word}\033[0m" for word in agent_words]
print(f"Selected target (agent) words: {', '.join(agent_words_display)}")

# Get normalized embeddings for the chosen combo
agent_embeddings = np.stack([normalized_vocab[w] for w in agent_words], axis=0)  # shape (num_agent_words, D)

# ---------------------------
# Agent group analysis (similarity & variance) for chosen combo
# ---------------------------
print("\n\033[1;33m----- AGENT GROUP ANALYSIS -----\033[0m")
K = agent_embeddings.shape[0]
# Pairwise cosine similarity (since normalized, dot = cosine)
pairwise = agent_embeddings @ agent_embeddings.T  # (K, K)
# Off-diagonal stats
mask = ~np.eye(K, dtype=bool)
off_diag = pairwise[mask]
max_cos = float(np.max(off_diag)) if off_diag.size > 0 else 1.0
min_cos = float(np.min(off_diag)) if off_diag.size > 0 else 1.0
mean_cos = float(np.mean(off_diag)) if off_diag.size > 0 else 1.0
std_cos = float(np.std(off_diag)) if off_diag.size > 0 else 0.0
print(f"Pairwise cosine (off-diagonal): max={max_cos:.3f}, min={min_cos:.3f}, mean={mean_cos:.3f}, std={std_cos:.3f}")
# Centroid cohesion
centroid = agent_embeddings.mean(axis=0)
centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
cos_to_centroid = agent_embeddings @ centroid_norm  # (K,)
ctc_mean = float(np.mean(cos_to_centroid))
ctc_std = float(np.std(cos_to_centroid))
ctc_min = float(np.min(cos_to_centroid))
ctc_max = float(np.max(cos_to_centroid))
print(f"Cosine to centroid: mean={ctc_mean:.3f}, std={ctc_std:.3f}, min={ctc_min:.3f}, max={ctc_max:.3f}")
# Intra-cluster variance proxy (on unit sphere): 1 - mean cosine to centroid
intra_var = float(1.0 - ctc_mean)
print(f"Intra-group variance (proxy): {intra_var:.3f}")
# ---------------------------

# Report top-5 candidates for the best combo
rank_indices = np.argsort(-best_margins_arr)
ranked = []
for idx in rank_indices[:max(5, len(rank_indices))]:  # keep for later printing
    ranked.append((best_candidate_words[idx], float(best_margins_arr[idx]), float(best_pos_min_arr[idx]), float(best_neg_max_arr[idx])))

print("\nTop 5 margin-ranked candidate code words:")
for i, (w, score, pos_s, neg_s) in enumerate(ranked[:5]):
    if i == 0:  # Top 1 - make it bold
        print(f"\033[1m{w}\033[0m: margin={score:.3f} (min_pos={pos_s:.3f}, max_neg={neg_s:.3f})")
    else:
        print(f"{w}: margin={score:.3f} (min_pos={pos_s:.3f}, max_neg={neg_s:.3f})")

# Choose the best candidate (already determined)
print(f"\nChosen code word: '\033[1m{best_word}\033[0m'")

print(f"\n\033[1;32m===== 🕵️ DECODER =====\033[0m")
print("Scoring similarity of the code word against board words...")

#########################
### @@ Decoder RUN @@ ###
#########################

comparison_words = agent_words + innocent_words + [w for w in agents if w not in agent_words + innocent_words]
similarities = {}
for w in comparison_words:
    similarities[w] = float(np.dot(best_vec, normalized_vocab[w.lower()]))  # cosine via dot (already normalized)

sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
print(f"\nComparing '{best_word}' to agent and innocent words:")

# Display top num_agent_words with special formatting
print(f"\n🎯 Top {num_agent_words} most similar words:")
print("=" * 40)
for i, (w, sim) in enumerate(sorted_similarities[:num_agent_words]):
    if w in agents:
        if w in agent_words:
            print(f"\033[1m\033[34m{w}\033[0m: {sim:.3f}")  # Bold blue
        else:
            print(f"\033[34m{w}\033[0m: {sim:.3f}")  # Blue
    elif w in agent_words:
        print(f"\033[1m{w}\033[0m: {sim:.3f}")  # Bold
    else:
        print(f"{w}: {sim:.3f}")  # Default

# Display remaining words if any
if len(sorted_similarities) > num_agent_words:
    print(f"\n📋 Remaining words:")
    print("-" * 25)
    for w, sim in sorted_similarities[num_agent_words:]:
        if w in agents:
            if w in agent_words:
                print(f"\033[1m\033[34m{w}\033[0m: {sim:.3f}")  # Bold blue
            else:
                print(f"\033[34m{w}\033[0m: {sim:.3f}")  # Blue
        elif w in agent_words:
            print(f"\033[1m{w}\033[0m: {sim:.3f}")  # Bold
        else:
            print(f"{w}: {sim:.3f}")  # Default
