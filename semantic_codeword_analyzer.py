import os
import pickle
import random
from typing import Dict, List, Tuple
from itertools import combinations
import time

import numpy as np
from utils import load_words


ANSI_RESET = "\033[0m"
ANSI_BLUE = "\033[34m"
ANSI_BOLD = "\033[1m"
ANSI_BOLD_BLUE = "\033[1;34m"
ANSI_MAGENTA_BOLD = "\033[1;35m"
ANSI_CYAN_BOLD = "\033[1;36m"
ANSI_YELLOW_BOLD = "\033[1;33m"


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class SemanticCodewordAnalyzer:
    """
    Runs the encoder/decoder semantic code word selection pipeline with analysis.

    Replicates the console output from main.py and stores all statistics
    as attributes for subsequent multi-run analysis.
    """

    def __init__(
        self,
        word_dataset_path: str = os.path.join("data", "simple_word_dataset.pkl"),
        vectorestore_path: str = os.path.join("data", "wordnet_bert_embedded_store.pkl"),
        seed: int | None = None,
    ) -> None:
        self.word_dataset_path = word_dataset_path
        self.vectorestore_path = vectorestore_path
        self.seed = seed

        # Outputs / stats
        self.game_agents: List[str] = []
        self.game_innocents: List[str] = []
        self.selected_agent_words: List[str] = []

        self.agent_pairwise_stats: Dict[str, float] = {}
        self.agent_centroid_stats: Dict[str, float] = {}
        self.intra_group_variance: float = 0.0

        self.top_candidates: List[Tuple[str, float, float, float]] = []  # (word, margin, min_pos, max_neg)
        self.best_word: str | None = None
        self.sorted_similarities: List[Tuple[str, float]] = []

        self.normalized_vocab: Dict[str, np.ndarray] = {}
        self.optimization_time: float = 0.0
        self.optimal_n: int = 0

        # confidence -> indicating what n_optimal to choose after reaching inflection point
        self.confidence = 'top' # 'floor', 'round' 

    def _load_vectorestore(self) -> None:
        with open(self.vectorestore_path, "rb") as f:
            vocab_embeddings = pickle.load(f)
        self.normalized_vocab = {w: l2_normalize(v) for w, v in vocab_embeddings.items()}

    def _sample_board(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)
        game_words = load_words(self.word_dataset_path)
        agents = [w.lower() for w in random.sample(game_words, 9)]
        innocents = [w.lower() for w in random.sample([w for w in game_words if w not in agents], 16)]
        self.game_agents = agents
        self.game_innocents = innocents

    def _print_board(self) -> None:
        agents_display = []
        for i, agent in enumerate(self.game_agents):
                agents_display.append(f"{ANSI_BLUE}{agent}{ANSI_RESET}")
        print(f"Agents: {', '.join(agents_display)}")
        print(f"Inocents: {', '.join(self.game_innocents)}")

    def _find_optimal_n_and_compute_best(self, innocent_words: List[str]) -> Tuple[List[str], str, List[Tuple[str, float, float, float]], int]:
        # Get valid agent candidate pool
        agent_candidate_pool = [w.lower() for w in self.game_agents]
        valid_agent_pool = [w for w in agent_candidate_pool if w in self.normalized_vocab]
        
        max_possible_n = min(9, len(valid_agent_pool))
        
        if max_possible_n < 1:
            raise ValueError("Not enough selected agent words exist in the embedding store.")

        # Innocent matrix for fast max similarity computation (shared across all n values)
        if len(innocent_words) == 0:
            raise ValueError("No innocent words found in the embedding store.")
        innoc_mat = np.stack([self.normalized_vocab[w] for w in innocent_words], axis=0)

        print(f"Vocab size: {len(self.normalized_vocab)}, Innocent words: {len(innocent_words)}")
        print(f"Agent candidates: {len(valid_agent_pool)}, Max possible n: {max_possible_n}")

        start_time = time.time()

        # OPTIMIZED VERSION: Precompute all matrices and use vectorized operations
        # Build full candidate pool (excluding board words)
        board_words = set(valid_agent_pool + innocent_words)
        candidate_words = [w for w in self.normalized_vocab.keys() if w not in board_words]
        print(f"Candidate words: {len(candidate_words)}")

        if len(candidate_words) == 0:
            raise ValueError("No candidate words available after excluding board words.")

        # Precompute candidate matrix (N x D)
        V = np.stack([self.normalized_vocab[w] for w in candidate_words], axis=0)

        # Precompute agent pool matrix (A x D) where A = number of agent candidates
        agent_pool_mat = np.stack([self.normalized_vocab[w] for w in valid_agent_pool], axis=0)

        # Precompute all pairwise similarities between candidates and agents (N x A)
        cand_agent_sims = V @ agent_pool_mat.T

        # Precompute all candidate-innocent similarities and take max (N,)
        cand_innoc_sims = V @ innoc_mat.T  # (N x M)
        neg_max_global = np.max(cand_innoc_sims, axis=1)  # (N,) - max innocent similarity per candidate

        # Analyze all possible n values to find optimal
        n_results = {}  # n -> (best_margin, best_combo, best_word, margins_arr, pos_min_arr)
        
        print(f"\n🔍 Analyzing all target counts (n=1 to {max_possible_n})...")
        
        for n in range(1, max_possible_n + 1):
            print(f"  Analyzing n={n}...")
            
            best_margin_for_n = -1e9
            best_combo_for_n = None
            best_word_for_n = None
            best_margins_arr_for_n = None
            best_pos_min_arr_for_n = None
            
            combo_count = 0
            for combo in combinations(range(len(valid_agent_pool)), n):
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
                
                # Update best for this n
                if local_best_margin > best_margin_for_n:
                    best_margin_for_n = local_best_margin
                    best_combo_for_n = [valid_agent_pool[i] for i in combo]
                    best_word_for_n = candidate_words[local_best_idx]
                    best_margins_arr_for_n = margins
                    best_pos_min_arr_for_n = pos_min
            
            n_results[n] = (best_margin_for_n, best_combo_for_n, best_word_for_n, best_margins_arr_for_n, best_pos_min_arr_for_n)
            print(f"    n={n}: best_margin={best_margin_for_n:.3f}, combinations={combo_count}")

        self.optimization_time = time.time() - start_time
        print(f"Optimization completed in {self.optimization_time:.2f} seconds")

        # Find optimal n using inflection point analysis
        optimal_n = self._calculate_inflection_point(n_results, polynomial_order=3)
        
        # Get results for optimal n
        best_margin, best_combo, best_word, best_margins_arr, best_pos_min_arr = n_results[optimal_n]
        
        print(f"\n📊 Optimal n selected: {optimal_n} (margin: {best_margin:.3f})")

        # Validate we found a best combo/word
        if best_combo is None or best_word is None:
            raise ValueError("Failed to find a valid code word for any agent word combination.")

        # Build ranked list from best combination results
        rank_indices = np.argsort(-best_margins_arr)
        ranked = []
        for idx in rank_indices[:max(5, len(rank_indices))]:
            ranked.append(
                (
                    candidate_words[idx],
                    float(best_margins_arr[idx]),
                    float(best_pos_min_arr[idx]),
                    float(neg_max_global[idx]),
                )
            )

        self.selected_agent_words = best_combo
        self.top_candidates = ranked
        self.best_word = best_word
        self.optimal_n = optimal_n
        return best_combo, best_word, ranked, optimal_n
    
    def _calculate_inflection_point(self, n_results: Dict[int, Tuple], polynomial_order: int = 5) -> int:
        """Calculate inflection point of margin curve to find optimal n"""
        
        # Extract margin data
        n_values = list(n_results.keys())
        margin_values = [n_results[n][0] for n in n_values]  # best_margin for each n
        
        print(f"\n📈 Margin analysis:")
        for n, margin in zip(n_values, margin_values):
            print(f"  n={n}: margin={margin:.3f}")
        
        if len(n_values) < 3:
            print(f"  🎯 Not enough data points, choosing n={max(n_values)}")
            return max(n_values)
        
        try:
            # Fit polynomial (limit order to avoid overfitting)
            effective_order = min(polynomial_order, len(n_values) - 1)
            coeffs = np.polyfit(n_values, margin_values, effective_order)
            
            print(f"  📊 Fitted polynomial of order {effective_order}")
            
            # For polynomial: margin = a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0
            # Second derivative: d²margin/dx² = n*(n-1)*a_n*x^(n-2) + (n-1)*(n-2)*a_(n-1)*x^(n-3) + ...
            
            # Calculate second derivative coefficients
            if len(coeffs) >= 3:
                second_deriv_coeffs = []
                for i, coeff in enumerate(coeffs[:-2]):  # exclude last 2 terms (linear and constant)
                    power = len(coeffs) - 1 - i  # original power
                    if power >= 2:
                        second_deriv_coeffs.append(coeff * power * (power - 1))
                
                if len(second_deriv_coeffs) > 0:
                    # Find roots of second derivative (inflection points)
                    if len(second_deriv_coeffs) == 1:
                        # Linear second derivative, no inflection point
                        inflection_points = []
                    else:
                        inflection_points = np.roots(second_deriv_coeffs)
                        # Keep only real inflection points within our range
                        inflection_points = [float(x.real) for x in inflection_points 
                                           if abs(x.imag) < 1e-10 and min(n_values) <= x.real <= max(n_values)]
                    
                    if inflection_points:
                        # Choose the first inflection point (where decline starts accelerating)
                        inflection_point = min(inflection_points)
                        # Three options for choosing optimal_n based on inflection point
                        if hasattr(self, 'confidence') and self.confidence == 'top':
                            optimal_n = max(1, min(max(n_values), int(np.floor(inflection_point))+1))
                        elif hasattr(self, 'confidence') and self.confidence == 'round':
                            optimal_n = max(1, min(max(n_values), int(np.round(inflection_point))))
                        else:  # default: 'floor' (more conservative)
                            optimal_n = max(1, min(max(n_values), int(np.floor(inflection_point))))
                        print(f"  📈 Inflection point at n={inflection_point:.2f} → choosing n={optimal_n}")
                        return optimal_n
            
            # Fallback: find where first derivative is most positive (shallowest decline)
            if len(coeffs) >= 2:
                first_deriv_coeffs = []
                for i, coeff in enumerate(coeffs[:-1]):  # exclude constant term
                    power = len(coeffs) - 1 - i  # original power
                    first_deriv_coeffs.append(coeff * power)
                
                # Evaluate first derivative at each n
                derivatives = []
                for n in n_values:
                    deriv = sum(coeff * (n ** (len(first_deriv_coeffs) - 1 - i)) 
                              for i, coeff in enumerate(first_deriv_coeffs))
                    derivatives.append((n, deriv))
                
                # Find where derivative is most positive (shallowest decline)
                max_deriv_n = max(derivatives, key=lambda x: x[1])[0]
                print(f"  📈 Shallowest decline at n={max_deriv_n} (derivative={max(derivatives, key=lambda x: x[1])[1]:.4f}) → choosing n={max_deriv_n}")
                return max_deriv_n
                
        except Exception as e:
            print(f"  ⚠️  Polynomial fitting failed ({e}), using fallback strategy")
        
        # Final fallback: choose n where margin is still reasonable compared to n=1
        margin_1 = margin_values[0] if len(margin_values) > 0 else 0
        
        for i, (n, margin) in enumerate(zip(n_values, margin_values)):
            if i == 0:  # n=1
                continue
            if margin < 0.7 * margin_1:  # significant drop
                optimal_n = n_values[i-1] if i > 0 else 1
                print(f"  🎯 Fallback: margin drop detected at n={n}, choosing n={optimal_n}")
                return optimal_n
        
        # Ultimate fallback
        optimal_n = min(2, max(n_values))
        print(f"  🎯 Ultimate fallback: choosing n={optimal_n}")
        return optimal_n

    def _agent_group_analysis(self, agent_embeddings: np.ndarray) -> None:
        K = agent_embeddings.shape[0]
        pairwise = agent_embeddings @ agent_embeddings.T
        mask = ~np.eye(K, dtype=bool)
        off_diag = pairwise[mask]
        max_cos = float(np.max(off_diag)) if off_diag.size > 0 else 1.0
        min_cos = float(np.min(off_diag)) if off_diag.size > 0 else 1.0
        mean_cos = float(np.mean(off_diag)) if off_diag.size > 0 else 1.0
        std_cos = float(np.std(off_diag)) if off_diag.size > 0 else 0.0
        self.agent_pairwise_stats = {
            "max": max_cos,
            "min": min_cos,
            "mean": mean_cos,
            "std": std_cos,
        }
        centroid = agent_embeddings.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
        cos_to_centroid = agent_embeddings @ centroid_norm
        ctc_mean = float(np.mean(cos_to_centroid))
        ctc_std = float(np.std(cos_to_centroid))
        ctc_min = float(np.min(cos_to_centroid))
        ctc_max = float(np.max(cos_to_centroid))
        self.agent_centroid_stats = {
            "mean": ctc_mean,
            "std": ctc_std,
            "min": ctc_min,
            "max": ctc_max,
        }
        self.intra_group_variance = float(1.0 - ctc_mean)



    def _print_encoder_section(self) -> None:
        # Encoder banner and selected targets
        agent_words_display = [f"{ANSI_BOLD}{w}{ANSI_RESET}" for w in self.selected_agent_words]
        print(f"\n{ANSI_MAGENTA_BOLD}===== 🎯 ENCODER ====={ANSI_RESET}")
        print(f"Optimal target count (n): {self.optimal_n}")
        print(f"Selected target (agent) words: {', '.join(agent_words_display)}")

        # Agent Group Analysis prints
        print(f"\n{ANSI_YELLOW_BOLD}----- AGENT GROUP ANALYSIS -----{ANSI_RESET}")
        ps = self.agent_pairwise_stats
        print(
            f"Pairwise cosine (off-diagonal): max={ps['max']:.3f}, min={ps['min']:.3f}, "
            f"mean={ps['mean']:.3f}, std={ps['std']:.3f}"
        )
        cs = self.agent_centroid_stats
        print(
            f"Cosine to centroid: mean={cs['mean']:.3f}, std={cs['std']:.3f}, "
            f"min={cs['min']:.3f}, max={cs['max']:.3f}"
        )
        print(f"Intra-group variance (proxy): {self.intra_group_variance:.3f}")

    def _print_candidates_and_choice(self) -> None:
        print("\nTop 5 margin-ranked candidate code words:")
        for w, score, pos_s, neg_s in self.top_candidates[:5]:
            print(f"{w}: margin={score:.3f} (min_pos={pos_s:.3f}, max_neg={neg_s:.3f})")
        print(f"\nChosen code word: '{self.best_word}'")

    def _print_decoder_section(self, similarities_sorted: List[Tuple[str, float]]) -> None:
        print(f"\n{ANSI_CYAN_BOLD}===== 🕵️ DECODER ====={ANSI_RESET}")
        print("Scoring similarity of the code word against board words...")
        for w, sim in similarities_sorted:
            word_type = "AGENT" if w in self.selected_agent_words else "INNOCENT"
            if w in self.game_agents:
                if w in self.selected_agent_words:
                    print(f"{ANSI_BOLD}{w}{ANSI_RESET} ({word_type}): {sim:.3f}")
                else:
                    print(f"{w}{ANSI_RESET} ({word_type}): {sim:.3f}")
            elif w in self.selected_agent_words:
                print(f"{w} ({word_type}): {sim:.3f}")
            else:
                print(f"{w} ({word_type}): {sim:.3f}")

    def _decoder_analysis(self, best_word: str, comparison_words: List[str], print_results: bool = True) -> List[Tuple[str, float]]:
        """Perform decoder analysis by computing similarities between the best word and board words."""
        best_vec = self.normalized_vocab[best_word]

        similarities = {w: float(np.dot(best_vec, self.normalized_vocab[w])) for w in comparison_words}
        self.sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        if print_results:
            self._print_decoder_section(self.sorted_similarities)

        return self.sorted_similarities

    def run(self) -> "SemanticCodewordAnalyzer":
        # Load vector store and board
        self._load_vectorestore()
        self._sample_board()
        self._print_board()

        # Innocents present in vocab
        innocent_words = [w for w in self.game_innocents if w in self.normalized_vocab]
        
        # Find optimal n and compute best candidate word
        agent_words, best_word, ranked, optimal_n = self._find_optimal_n_and_compute_best(innocent_words)

        # Encoder analysis
        agent_embeddings = np.stack([self.normalized_vocab[w] for w in agent_words], axis=0)
        self._agent_group_analysis(agent_embeddings)
        self._print_encoder_section()

        # Print candidates and choice
        self._print_candidates_and_choice()

        # Decoder analysis
        self._decoder_analysis(best_word, agent_words + innocent_words)

        return self


if __name__ == "__main__":
    vectorestore_path=os.path.join("data", "wordnet_openai_embedded_store.pkl")
    analyzer = SemanticCodewordAnalyzer(vectorestore_path=vectorestore_path)
    analyzer.run() 