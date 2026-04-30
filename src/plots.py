import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_PATH = "results/experiment_results.csv"
OUTPUT_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

# 1. Integrality Gap vs n (Complete Graphs)
df_complete = df[df["family"] == "complete"]

plt.figure()
plt.plot(df_complete["n"], df_complete["integrality_gap"], marker='o')
plt.xlabel("Number of vertices (n)")
plt.ylabel("Integrality Gap (ILP / LP)")
plt.title("Integrality Gap for Complete Graphs")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/complete_gap.png")
plt.close()


# 2. Random graphs: ratio vs density
df_random = df[df["family"] == "random"]

grouped = df_random.groupby("param_p").mean(numeric_only=True)

plt.figure()
plt.plot(grouped.index, grouped["rounded_vs_ilp_ratio"], marker='o', label="Rounding")
plt.plot(grouped.index, grouped["greedy_vs_ilp_ratio"], marker='o', label="Greedy")
plt.xlabel("Edge Probability (p)")
plt.ylabel("Approximation Ratio")
plt.title("Approximation Ratio vs Density")
plt.legend()
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/random_density.png")
plt.close()


# 3. Greedy vs Rounding scatter
df_valid = df.dropna(subset=["rounded_vs_ilp_ratio", "greedy_vs_ilp_ratio"])

plt.figure()
plt.scatter(df_valid["rounded_vs_ilp_ratio"], df_valid["greedy_vs_ilp_ratio"], alpha=0.6)
plt.xlabel("Rounding / ILP")
plt.ylabel("Greedy / ILP")
plt.title("Greedy vs Rounding Performance")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/greedy_vs_rounding.png")
plt.close()


# 4. Runtime comparison
runtime_cols = ["lp_runtime_sec", "rounding_runtime_sec", "greedy_runtime_sec"]

runtime_means = df[runtime_cols].mean()

plt.figure()
runtime_means.plot(kind='bar')
plt.ylabel("Average Runtime (seconds)")
plt.title("Runtime Comparison")
plt.xticks(rotation=30)
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/runtime.png")
plt.close()

print("Plots saved in:", OUTPUT_DIR)