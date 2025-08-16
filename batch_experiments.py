import os
import argparse
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from semantic_codeword_analyzer import SemanticCodewordAnalyzer


def run_batch(
    k_min: int = 2,
    k_max: int = 9,
    runs_per_k: int = 10,
    out_csv: str = os.path.join("data", "bert_semantic_runs.csv"),
    seed_base: int = 42,
    suppress_logs: bool = True,
    vectorestore_path: str = os.path.join("data", "wordnet_bert_embedded_store.pkl"),
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    total_runs = (k_max - k_min + 1) * runs_per_k
    with tqdm(total=total_runs, desc="Running experiments") as pbar:
        for k in range(k_min, k_max + 1):
            for i in range(runs_per_k):
                seed = seed_base + i
                analyzer = SemanticCodewordAnalyzer(num_agent_words=k, seed=seed, vectorestore_path=vectorestore_path)
                if suppress_logs:
                    with redirect_stdout(io.StringIO()):
                        analyzer.run()
                else:
                    analyzer.run()

                # Extract stats for row
                row: Dict[str, Any] = {
                    "num_agent_words": k,
                    "seed": seed,
                    "best_word": analyzer.best_word,
                    # Pairwise stats
                    "pairwise_max": analyzer.agent_pairwise_stats.get("max"),
                    "pairwise_min": analyzer.agent_pairwise_stats.get("min"),
                    "pairwise_mean": analyzer.agent_pairwise_stats.get("mean"),
                    "pairwise_std": analyzer.agent_pairwise_stats.get("std"),
                    # Centroid stats
                    "centroid_mean": analyzer.agent_centroid_stats.get("mean"),
                    "centroid_std": analyzer.agent_centroid_stats.get("std"),
                    "centroid_min": analyzer.agent_centroid_stats.get("min"),
                    "centroid_max": analyzer.agent_centroid_stats.get("max"),
                    # Variance proxy
                    "intra_group_variance": analyzer.intra_group_variance,
                    # Candidate stats for chosen best
                    "best_margin": None,
                    "best_min_pos": None,
                    "best_max_neg": None,
                    # Board composition counts
                    "n_agents_total": len(analyzer.game_agents),
                    "n_innocents_total": len(analyzer.game_innocents),
                    "n_agents_selected": len(analyzer.selected_agent_words),
                }

                if analyzer.top_candidates:
                    bw = analyzer.best_word
                    for w, margin, min_pos, max_neg in analyzer.top_candidates:
                        if w == bw:
                            row["best_margin"] = margin
                            row["best_min_pos"] = min_pos
                            row["best_max_neg"] = max_neg
                            break

                # Decoder results: top-k predictions and match against agent words
                if analyzer.sorted_similarities:
                    pred_top_k = [w for (w, _) in analyzer.sorted_similarities[:k]]
                    match_words = [w for w in pred_top_k if w in analyzer.selected_agent_words]
                    # Rank (1-based) of first correct match, 0 if none
                    first_match_rank = 0
                    for idx, w in enumerate(pred_top_k):
                        if w in analyzer.selected_agent_words:
                            first_match_rank = idx + 1
                            break
                    # Leading consecutive correct matches from top-1
                    leading_matches_prefix = 0
                    for w in pred_top_k:
                        if w in analyzer.selected_agent_words:
                            leading_matches_prefix += 1
                        else:
                            break
                    row["decoder_top_k"] = pred_top_k
                    row["decoder_top_k_str"] = "|".join(pred_top_k)
                    row["matches_count"] = len(match_words)
                    row["matches_rate"] = (len(match_words) / k) if k > 0 else 0.0
                    row["matches_words"] = match_words
                    row["matches_words_str"] = "|".join(match_words)
                    row["selected_agent_words"] = analyzer.selected_agent_words
                    row["selected_agent_words_str"] = "|".join(analyzer.selected_agent_words)
                    row["first_match_rank"] = first_match_rank
                    row["leading_matches_prefix"] = leading_matches_prefix

                rows.append(row)
                pbar.update(1)

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    return df



if __name__ == "__main__":
    # Configuration variables
    k_min = 2
    k_max = 9
    runs_per_k = 10
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = "roberta" # "gemini", "bert", "roberta", "openai"
    out_csv = os.path.join("data", f"{model}_semantic_runs_{timestamp}.csv")
    vectorestore_path = os.path.join("data", f"wordnet_{model}_embedded_store.pkl")
    seed_base = 42
    show_logs = False
    

    df = run_batch(
        k_min=k_min,
        k_max=k_max,
        runs_per_k=runs_per_k,
        out_csv=out_csv,
        seed_base=seed_base,
        suppress_logs=not show_logs,
        vectorestore_path=vectorestore_path,
    )

    print(f"Total runs: {len(df)}")
    print(df.head(20))
    print(f"Saved results to: {out_csv}")
