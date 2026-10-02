from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.aggregate import load_results_for_k
from utils import load_config
from utils.logger import setup_logger

log = setup_logger("make_plots")


def load_all_k(results_dir: str, k_values: list[int]) -> pd.DataFrame:
    """Concatenate per-k generation result files into one long-format dataframe."""
    frames = [load_results_for_k(results_dir, k) for k in k_values]
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """Return the percentile bootstrap confidence interval for the sample mean.

    Args:
        values: Observed metric values.
        n_boot: Number of bootstrap resamples.
        seed: Seed used to make resampling reproducible.
        alpha: Total probability excluded from the interval's tails.

    Returns:
        The lower and upper confidence bounds, or ``(nan, nan)`` for no values.
    """
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, 100 * (alpha / 2))
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (lo, hi)


def build_plot_data(df: pd.DataFrame, k_values: list[int], n_boot: int, seed: int) -> pd.DataFrame:
    """Calculate metric means and bootstrap intervals for each retrieval depth.

    Uses the result schema produced by ``generate.py`` and includes overall F1,
    exact match, Recall@k, and F1 conditional on retrieving the gold passage.
    """
    records = []
    for k in k_values:
        g = df[df["k"] == k]
        f1_vals = g["f1"].to_numpy(dtype=float)
        em_vals = g["exact_match"].to_numpy(dtype=float)
        retrieved_mask = g["gold_retrieved"].astype(bool)

        f1_mean = f1_vals.mean()
        em_mean = em_vals.mean()
        recall_at_k = retrieved_mask.mean()
        f1_lo, f1_hi = bootstrap_ci(f1_vals, n_boot=n_boot, seed=seed)
        em_lo, em_hi = bootstrap_ci(em_vals, n_boot=n_boot, seed=seed)

        cond_f1_vals = g.loc[retrieved_mask, "f1"].to_numpy(dtype=float)
        cond_f1_mean = cond_f1_vals.mean() if len(cond_f1_vals) else np.nan
        cond_f1_lo, cond_f1_hi = bootstrap_ci(cond_f1_vals, n_boot=n_boot, seed=seed)

        records.append({
            "k": k,
            "recall_at_k": recall_at_k,
            "f1": f1_mean,
            "f1_ci_lo": f1_lo,
            "f1_ci_hi": f1_hi,
            "em": em_mean,
            "em_ci_lo": em_lo,
            "em_ci_hi": em_hi,
            "conditional_f1_given_retrieved": cond_f1_mean,
            "conditional_f1_ci_lo": cond_f1_lo,
            "conditional_f1_ci_hi": cond_f1_hi,
        })
    return pd.DataFrame(records).sort_values("k").reset_index(drop=True)


def plot_recall_f1_vs_k(summary: pd.DataFrame, outpath: str):
    """Plot Recall@k, F1, and exact match against retrieval depth and save the figure."""
    fig, ax1 = plt.subplots(figsize=(6, 4.2), dpi=300)

    color_f1 = "#1f77b4"
    color_em = "#2ca02c"
    color_recall = "#d62728"

    ax1.set_xlabel("Retrieval depth (k)")
    ax1.set_ylabel("Answer score (F1 / EM)")
    ax1.plot(summary["k"], summary["f1"], marker="o", color=color_f1, label="F1")
    ax1.fill_between(summary["k"], summary["f1_ci_lo"], summary["f1_ci_hi"],
                     color=color_f1, alpha=0.15)
    ax1.plot(summary["k"], summary["em"], marker="^", color=color_em, label="EM")
    ax1.fill_between(summary["k"], summary["em_ci_lo"], summary["em_ci_hi"],
                     color=color_em, alpha=0.15)
    ax1.set_xticks(summary["k"])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Recall@k", color=color_recall)
    ax2.plot(summary["k"], summary["recall_at_k"], marker="s", linestyle="--",
             color=color_recall, label="Recall@k")
    ax2.tick_params(axis="y", labelcolor=color_recall)
    ax2.set_ylim(0, 1.02)

    fig.suptitle("Retrieval coverage vs. answer quality")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_conditional_f1_vs_k(summary: pd.DataFrame, outpath: str):
    """Plot overall and gold-retrieval-conditional F1 against depth and save the figure."""
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

    ax.plot(summary["k"], summary["f1"], marker="o", label="F1 (all questions)",
            color="#1f77b4")
    ax.fill_between(summary["k"], summary["f1_ci_lo"], summary["f1_ci_hi"],
                    color="#1f77b4", alpha=0.12)

    ax.plot(summary["k"], summary["conditional_f1_given_retrieved"], marker="^",
            label="F1 | gold passage retrieved", color="#2ca02c")
    ax.fill_between(summary["k"], summary["conditional_f1_ci_lo"], summary["conditional_f1_ci_hi"],
                    color="#2ca02c", alpha=0.12)

    ax.set_xlabel("Retrieval depth (k)")
    ax.set_ylabel("F1")
    ax.set_xticks(summary["k"])
    ax.set_title("Does more context help, even with the right evidence retrieved?")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    """Load configured result files, compute plot data, and save both figures."""
    cfg = load_config()

    outdir = cfg["paths"]["main_results_dir"]
    results_dir = cfg["paths"]["main_results_dir"]

    Path(outdir).mkdir(parents=True, exist_ok=True)

    k_values = cfg["retrieval"]["k_values"]
    df = load_all_k(results_dir, k_values)

    n_boot = cfg["evaluation"]["bootstrap"]["n_boot"]
    seed = cfg["evaluation"]["bootstrap"]["seed"]
    plot_data = build_plot_data(df, k_values, n_boot=n_boot, seed=seed)

    fig1_path = str(Path(outdir) / "recall_f1_vs_k.png")
    plot_recall_f1_vs_k(plot_data, fig1_path)
    log.info(f"Saved figure -> {fig1_path}")

    fig2_path = str(Path(outdir) / "conditional_f1_vs_k.png")
    plot_conditional_f1_vs_k(plot_data, fig2_path)
    log.info(f"Saved figure -> {fig2_path}")


if __name__ == "__main__":
    main()