"""
Visualization module for synthetic data evaluation.

Provides reusable plotting functions for comparing real vs synthetic
distributions. All plots return matplotlib Figure objects so they can
be displayed inline in notebooks or saved to disk for reports.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS

# ============================================================
# STYLE CONFIGURATION
# ============================================================

REAL_COLOR = "#2E86AB"
SYNTHETIC_COLOR = "#E63946"
DIFF_CMAP = "RdBu_r"
CORR_CMAP = "coolwarm"


def _save_or_show(fig: plt.Figure, save_path: Optional[Path] = None, dpi: int = 150) -> plt.Figure:
    """
    Internal helper: save figure to disk if path provided, else just return it.
    """
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ============================================================
# 1. CONTINUOUS VARIABLES - HISTOGRAMS
# ============================================================

def plot_continuous_distributions(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: List[str] = None,
    wasserstein_results: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot overlapping histograms for continuous variables (real vs synthetic).

    Parameters
    ----------
    real : pd.DataFrame
        Real dataset.
    synthetic : pd.DataFrame
        Synthetic dataset.
    columns : list, optional
        Columns to plot. Defaults to CONTINUOUS_COLUMNS.
    wasserstein_results : pd.DataFrame, optional
        Output of compute_wasserstein() to annotate each subplot.
    save_path : Path, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
    """
    if columns is None:
        columns = CONTINUOUS_COLUMNS

    n = len(columns)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        real_vals = real[col].values
        synth_vals = synthetic[col].values

        min_val = min(real_vals.min(), synth_vals.min())
        max_val = max(real_vals.max(), synth_vals.max())
        bins = np.linspace(min_val, max_val, 40)

        ax.hist(real_vals, bins=bins, alpha=0.6, label="Real", color=REAL_COLOR, density=True)
        ax.hist(synth_vals, bins=bins, alpha=0.6, label="Synthetic", color=SYNTHETIC_COLOR, density=True)

        # Annotate with Wasserstein if provided
        title = col
        if wasserstein_results is not None:
            row = wasserstein_results[wasserstein_results["variable"] == col]
            if not row.empty:
                w_pct = row.iloc[0]["wasserstein_pct"]
                title = f"{col}\nWasserstein rel: {w_pct:.2f}%"

        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return _save_or_show(fig, save_path)


# ============================================================
# 2. CATEGORICAL VARIABLES - GROUPED BARS
# ============================================================

def plot_categorical_distributions(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: List[str],
    jsd_results: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot grouped bar charts for categorical variables (real vs synthetic).

    For variables with cardinality > top_n, only the top categories
    by real frequency are displayed.

    Parameters
    ----------
    real : pd.DataFrame
        Real dataset.
    synthetic : pd.DataFrame
        Synthetic dataset.
    columns : list
        Categorical columns to plot.
    jsd_results : pd.DataFrame, optional
        Output of compute_jsd() to annotate each subplot.
    top_n : int
        Maximum categories to show per variable.
    save_path : Path, optional
        Save figure to this path.

    Returns
    -------
    plt.Figure
    """
    n = len(columns)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]

        real_dist = real[col].value_counts(normalize=True).sort_values(ascending=False)
        synth_dist = synthetic[col].value_counts(normalize=True)

        if real[col].nunique() > top_n:
            top_cats = real_dist.head(top_n).index
            real_dist = real_dist.loc[top_cats]
            synth_dist = synth_dist.reindex(top_cats, fill_value=0)
            suffix = f" (Top {top_n})"
        else:
            synth_dist = synth_dist.reindex(real_dist.index, fill_value=0)
            suffix = ""

        x = np.arange(len(real_dist))
        width = 0.4

        ax.bar(x - width / 2, real_dist.values, width, label="Real", color=REAL_COLOR, alpha=0.8)
        ax.bar(x + width / 2, synth_dist.values, width, label="Synthetic", color=SYNTHETIC_COLOR, alpha=0.8)

        title = f"{col}{suffix}"
        if jsd_results is not None:
            row = jsd_results[jsd_results["variable"] == col]
            if not row.empty:
                jsd_val = row.iloc[0]["jsd"]
                title = f"{title}\nJSD: {jsd_val:.4f}"

        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(real_dist.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Proportion")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return _save_or_show(fig, save_path)


# ============================================================
# 3. CORRELATION HEATMAPS (real, synthetic, difference)
# ============================================================

def plot_correlation_comparison(
    correlation_results: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot three side-by-side heatmaps: real, synthetic, and difference.

    Parameters
    ----------
    correlation_results : dict
        Output of compute_correlation_difference().
        Must contain keys: 'corr_real', 'corr_synthetic', 'difference',
        'frobenius_relative_pct'.
    save_path : Path, optional
        Save figure to this path.

    Returns
    -------
    plt.Figure
    """
    corr_real = correlation_results["corr_real"]
    corr_synth = correlation_results["corr_synthetic"]
    diff = correlation_results["difference"]
    frobenius_pct = correlation_results["frobenius_relative_pct"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(corr_real, annot=True, fmt=".2f", cmap=CORR_CMAP, vmin=-1, vmax=1,
                ax=axes[0], cbar_kws={"shrink": 0.7})
    axes[0].set_title("Correlation - Real Data", fontsize=12, weight="bold")

    sns.heatmap(corr_synth, annot=True, fmt=".2f", cmap=CORR_CMAP, vmin=-1, vmax=1,
                ax=axes[1], cbar_kws={"shrink": 0.7})
    axes[1].set_title("Correlation - Synthetic Data", fontsize=12, weight="bold")

    sns.heatmap(diff, annot=True, fmt=".2f", cmap=DIFF_CMAP, vmin=-0.3, vmax=0.3,
                ax=axes[2], cbar_kws={"shrink": 0.7})
    axes[2].set_title(f"Difference (Real - Synthetic)\nFrobenius rel: {frobenius_pct:.2f}%",
                      fontsize=12, weight="bold")

    plt.tight_layout()
    return _save_or_show(fig, save_path)


# ============================================================
# 4. V1 vs V2 COMPARISON
# ============================================================

def plot_zero_inflated_comparison(
    real: pd.DataFrame,
    synthetic_v1: pd.DataFrame,
    synthetic_v2: pd.DataFrame,
    columns: List[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare zero-inflated variable distributions: Real vs V1 vs V2.

    Designed to visually demonstrate whether the V2 log-transform
    improvement preserves the mass at zero better than V1.

    Parameters
    ----------
    real : pd.DataFrame
        Real dataset.
    synthetic_v1 : pd.DataFrame
        Synthetic data from V1 baseline (no log transform).
    synthetic_v2 : pd.DataFrame
        Synthetic data from V2 improved (with log transform).
    columns : list, optional
        Zero-inflated columns to compare. Defaults to ['capital.gain', 'capital.loss'].
    save_path : Path, optional
        Save figure to this path.

    Returns
    -------
    plt.Figure
    """
    if columns is None:
        columns = ["capital.gain", "capital.loss"]

    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        ax = axes[i]

        real_zero_pct = (real[col] == 0).mean() * 100
        v1_zero_pct = (synthetic_v1[col] == 0).mean() * 100
        v2_zero_pct = (synthetic_v2[col] == 0).mean() * 100

        labels = ["Real", "V1 baseline", "V2 improved"]
        values = [real_zero_pct, v1_zero_pct, v2_zero_pct]
        colors = [REAL_COLOR, "#F4A261", "#2A9D8F"]

        bars = ax.bar(labels, values, color=colors, alpha=0.85)
        ax.set_title(f"{col}: % of zero values", fontsize=12, weight="bold")
        ax.set_ylabel("% zeros")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3, axis="y")

        # Annotate bars with values
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                    f"{val:.1f}%", ha="center", fontsize=10, weight="bold")

    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_metrics_comparison_v1_v2(
    metrics_v1: dict,
    metrics_v2: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of TSTR ratios for V1 and V2.

    Parameters
    ----------
    metrics_v1 : dict
        Output of evaluate_all() for V1.
    metrics_v2 : dict
        Output of evaluate_all() for V2.
    save_path : Path, optional
        Save figure to this path.

    Returns
    -------
    plt.Figure
    """
    tstr_v1 = metrics_v1["tstr"]
    tstr_v2 = metrics_v2["tstr"]

    fig, ax = plt.subplots(figsize=(10, 5))

    metrics = tstr_v1["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, tstr_v1["ratio"], width, label="V1 baseline", color="#F4A261", alpha=0.85)
    bars2 = ax.bar(x + width / 2, tstr_v2["ratio"], width, label="V2 improved", color="#2A9D8F", alpha=0.85)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Perfect (= real)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("TSTR / Baseline ratio")
    ax.set_title("TSTR Utility Ratio: V1 vs V2", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    return _save_or_show(fig, save_path)