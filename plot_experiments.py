import os
import argparse
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float('nan')
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def plot_line_with_error(ax, x, y_mean, y_std=None, title="", xlabel="", ylabel=""):
    ax.plot(x, y_mean, marker='o', label='mean')
    if y_std is not None:
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, label='±1 std')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def run_plots(csv_path: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Group by num_agent_words
    g = df.groupby("num_agent_words")
    k_values = sorted(g.groups.keys())

    mean_matches_rate = g["matches_rate"].mean()
    std_matches_rate = g["matches_rate"].std()

    mean_first_rank = g["first_match_rank"].mean()
    std_first_rank = g["first_match_rank"].std()

    mean_prefix = g["leading_matches_prefix"].mean()
    std_prefix = g["leading_matches_prefix"].std()

    mean_variance = g["intra_group_variance"].mean()
    std_variance = g["intra_group_variance"].std()

    # Also compute mean matches_count
    mean_matches_count = g["matches_count"].mean()
    std_matches_count = g["matches_count"].std()

    # 1) Line plots grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    plot_line_with_error(
        axes[0], k_values, mean_matches_rate.loc[k_values].values, std_matches_rate.loc[k_values].values,
        title="Mean matches_rate vs num_agent_words", xlabel="num_agent_words", ylabel="matches_rate"
    )

    plot_line_with_error(
        axes[1], k_values, mean_first_rank.loc[k_values].values, std_first_rank.loc[k_values].values,
        title="Mean first_match_rank vs num_agent_words", xlabel="num_agent_words", ylabel="first_match_rank"
    )

    plot_line_with_error(
        axes[2], k_values, mean_prefix.loc[k_values].values, std_prefix.loc[k_values].values,
        title="Mean leading_matches_prefix vs num_agent_words", xlabel="num_agent_words", ylabel="leading_matches_prefix"
    )

    plot_line_with_error(
        axes[3], k_values, mean_variance.loc[k_values].values, std_variance.loc[k_values].values,
        title="Mean intra_group_variance vs num_agent_words", xlabel="num_agent_words", ylabel="intra_group_variance"
    )

    fig.tight_layout()
    line_path = os.path.join(outdir, "lineplots_summary.png")
    fig.savefig(line_path, dpi=150)

    # Additional line plot: mean matches_count vs num_agent_words
    fig_extra, ax_extra = plt.subplots(figsize=(7, 5))
    plot_line_with_error(
        ax_extra,
        k_values,
        mean_matches_count.loc[k_values].values,
        std_matches_count.loc[k_values].values,
        title="Mean matches_count vs num_agent_words",
        xlabel="num_agent_words",
        ylabel="matches_count",
    )
    extra_path = os.path.join(outdir, "line_mean_matches_count.png")
    fig_extra.savefig(extra_path, dpi=150)

    # 2) Scatter: num_agent_words vs leading_matches_prefix
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    x = df["num_agent_words"].values
    y = df["leading_matches_prefix"].values
    r = safe_corr(x, y)
    ax2.scatter(x, y, alpha=0.5)
    # Overlay mean line per num_agent_words
    ax2.plot(k_values, mean_prefix.loc[k_values].values, color='red', marker='o', linewidth=2, label='mean')
    ax2.set_title(f"num_agent_words vs leading_matches_prefix (r={r:.3f})")
    ax2.set_xlabel("num_agent_words")
    ax2.set_ylabel("leading_matches_prefix")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    scatter1_path = os.path.join(outdir, "scatter_k_vs_prefix.png")
    fig2.savefig(scatter1_path, dpi=150)

    # 3) Scatter: leading_matches_prefix vs intra_group_variance (with 5th-order trend)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    x2 = df["leading_matches_prefix"].values
    y2 = df["intra_group_variance"].values
    r2 = safe_corr(x2, y2)
    ax3.scatter(x2, y2, alpha=0.5)
    # Add 5th-order polynomial trend line
    mask = np.isfinite(x2) & np.isfinite(y2)
    if mask.sum() >= 6:  # need enough points for degree-3 fit
        xs = x2[mask]
        ys = y2[mask]
        coeffs = np.polyfit(xs, ys, 3)
        x_line = np.linspace(xs.min(), xs.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        ax3.plot(x_line, y_line, color='red', linewidth=2, label='3th-order trend')
        ax3.legend()
    ax3.set_title(f"leading_matches_prefix vs intra_group_variance (r={r2:.3f})")
    ax3.set_xlabel("leading_matches_prefix")
    ax3.set_ylabel("intra_group_variance")
    ax3.grid(True, alpha=0.3)
    scatter2_path = os.path.join(outdir, "scatter_prefix_vs_variance.png")
    fig3.savefig(scatter2_path, dpi=150)

    # Show all
    print("Saved plots:\n- {}\n- {}\n- {}\n- {}".format(line_path, extra_path, scatter1_path, scatter2_path))
    plt.show()


def run_plots_multi(indir: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    files = glob.glob(os.path.join(indir, "*_semantic_runs_*.csv"))
    if not files:
        print(f"No *_semantic_runs.csv files found in {indir}")
        return

    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    # Metrics to plot: (column, title, ylabel)
    metrics = [
        ("matches_rate", "Mean matches_rate vs num_agent_words", "matches_rate"),
        ("first_match_rank", "Mean first_match_rank vs num_agent_words", "first_match_rank"),
        ("leading_matches_prefix", "Mean leading_matches_prefix vs num_agent_words", "leading_matches_prefix"),
        ("intra_group_variance", "Mean intra_group_variance vs num_agent_words", "intra_group_variance"),
    ]

    for csv_path in files:
        model_name = os.path.basename(csv_path).replace("_semantic_runs.csv", "")
        df = pd.read_csv(csv_path)
        g = df.groupby("num_agent_words")
        k_values = sorted(g.groups.keys())
        for ax, (col, title, ylabel) in zip(axes, metrics):
            mean_vals = g[col].mean()
            ax.plot(k_values, mean_vals.loc[k_values].values, marker='o', label=model_name)
            ax.set_title(title)
            ax.set_xlabel("num_agent_words")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    out_path = os.path.join(outdir, "lineplots_summary_multi.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved multi-model lineplots to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot semantic experiments results")
    parser.add_argument("--csv", type=str, default=os.path.join("data", "semantic_runs.csv"))
    parser.add_argument("--outdir", type=str, default=os.path.join("data", "plots"))
    parser.add_argument("--multi_dir", type=str, default=None, help="Directory containing *_semantic_runs.csv to aggregate")
    args = parser.parse_args()

    if args.multi_dir:
        run_plots_multi(args.multi_dir, args.outdir)
        return

    run_plots(args.csv, args.outdir)


def plot_game_results(csv_path: str, output_dir: str = None) -> None:
    """
    Plot game results from AI vs AI Codenames experiments.
    
    Creates two subplots:
    1. Number of winning games by model
    2. Total points scored by model
    
    Args:
        csv_path: Path to game results CSV file
        output_dir: Directory to save plots (optional)
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} game results from {csv_path}")
    
    # Get all unique models
    all_models = set(df['red_model'].unique()) | set(df['blue_model'].unique())
    all_models = sorted(list(all_models))
    print(f"Models found: {all_models}")
    
    # Initialize tracking dictionaries
    model_wins = defaultdict(int)
    model_points = defaultdict(int)
    model_games = defaultdict(int)
    
    # Process each game result
    for _, row in df.iterrows():
        red_model = row['red_model']
        blue_model = row['blue_model']
        winner = row['winner']
        red_score = row['red_score']
        blue_score = row['blue_score']
        
        # Count games played
        model_games[red_model] += 1
        model_games[blue_model] += 1
        
        # Count wins
        if winner == red_model:
            model_wins[red_model] += 1
        elif winner == blue_model:
            model_wins[blue_model] += 1
        # Note: draws don't count as wins for either model
        
        # Count points (words found)
        model_points[red_model] += red_score
        model_points[blue_model] += blue_score
    
    # Convert to lists for plotting
    models = all_models
    wins = [model_wins[model] for model in models]
    points = [model_points[model] for model in models]
    games_played = [model_games[model] for model in models]
    win_rates = [model_wins[model] / model_games[model] * 100 if model_games[model] > 0 else 0 for model in models]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Number of wins
    bars1 = ax1.bar(models, wins, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)], alpha=0.7)
    ax1.set_title('Number of Winning Games by Model', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Number of Wins')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, win, games in zip(bars1, wins, games_played):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{win}\n({win/games*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10)
    
    # Subplot 2: Total points
    bars2 = ax2.bar(models, points, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)], alpha=0.7)
    ax2.set_title('Total Points Scored by Model', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Total Points (Words Found)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, point, games in zip(bars2, points, games_played):
        height = bar.get_height()
        avg_points = point / games if games > 0 else 0
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(points) * 0.01,
                f'{point}\n({avg_points:.1f}/game)', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n📊 Game Results Summary:")
    print(f"{'Model':<10} {'Games':<6} {'Wins':<5} {'Win%':<6} {'Points':<7} {'Avg Points':<10}")
    print("-" * 60)
    for model in models:
        games = model_games[model]
        wins_count = model_wins[model]
        win_pct = wins_count / games * 100 if games > 0 else 0
        total_points = model_points[model]
        avg_points = total_points / games if games > 0 else 0
        print(f"{model:<10} {games:<6} {wins_count:<5} {win_pct:<6.1f} {total_points:<7} {avg_points:<10.2f}")
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"game_results_analysis_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n💾 Plot saved to: {filepath}")
    
    plt.show()


def main_game_results():
    """Main function for plotting game results"""
    parser = argparse.ArgumentParser(description='Plot AI vs AI Codenames game results')
    parser.add_argument('csv_path', help='Path to game results CSV file')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots')
    
    args = parser.parse_args()
    
    plot_game_results(args.csv_path, args.output_dir)


if __name__ == "__main__":
    # Uncomment to plot specific game results
    # plot_game_results("data/game_results_20250814_014937.csv", "data/game_plots")
    
    # Or run with command line arguments
    # if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
    main_game_results()
    # else:
        # # Run original main() if needed 
        # run_plots_multi(os.path.join("data"), os.path.join("data", "plots"))

# if __name__ == "__main__":#     # Allow running without CLI by setting variables here
#     # if len(sys.argv) == 1:
#     #     model = "gemini" # "gemini", "bert", "roberta", "openai"
#     #     chosen_csv = os.path.join("data", f"{model}_semantic_runs_20250813_222524.csv")
#     #     chosen_outdir = os.path.join("data", f"{model}_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
#     #     run_plots(chosen_csv, chosen_outdir)
#     # else:
#     #     main()

