import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Load all CSV files matching the pattern
csv_files = glob.glob(os.path.join("data", "*_semantic_runs_*.csv"))

# Create subplots for original data, first derivative, and second derivative
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for csv_path in csv_files:
    model_name = os.path.basename(csv_path).replace("_semantic_runs", "").split("_")[0]
    df = pd.read_csv(csv_path)
    
    # Original scatter plot
    axes[0].scatter(df["num_agent_words"], df["best_margin"], alpha=0.6, label=model_name)
    
    # Add 5th-order polynomial trend line
    x = df["num_agent_words"].values
    y = df["best_margin"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 6:  # need enough points for degree-5 fit
        xs = x[mask]
        ys = y[mask]
        coeffs = np.polyfit(xs, ys, 5)
        x_line = np.linspace(xs.min(), xs.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        axes[0].plot(x_line, y_line, linewidth=2, label=f'{model_name} trend')
        
        # First derivative
        first_deriv_coeffs = np.polyder(coeffs, 1)
        y_first_deriv = np.polyval(first_deriv_coeffs, x_line)
        axes[1].plot(x_line, y_first_deriv, linewidth=2, label=f'{model_name} 1st deriv')
        
        # Second derivative
        second_deriv_coeffs = np.polyder(coeffs, 2)
        y_second_deriv = np.polyval(second_deriv_coeffs, x_line)
        axes[2].plot(x_line, y_second_deriv, linewidth=2, label=f'{model_name} 2nd deriv')

# Configure first subplot (original data)
axes[0].set_xlabel("num_agent_words")
axes[0].set_ylabel("best_margin")
axes[0].set_title("Best Margin vs Number of Agent Words")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Configure second subplot (first derivative)
axes[1].set_xlabel("num_agent_words")
axes[1].set_ylabel("d(best_margin)/d(num_agent_words)")
axes[1].set_title("First Derivative of Best Margin")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Configure third subplot (second derivative)
axes[2].set_xlabel("num_agent_words")
axes[2].set_ylabel("d²(best_margin)/d(num_agent_words)²")
axes[2].set_title("Second Derivative of Best Margin")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()
