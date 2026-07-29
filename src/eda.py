"""
eda.py — Exploratory Data Analysis functions for the Crop Recommendation dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Colour palette ──────────────────────────────────────────────────────────
BG       = "#0d1117"
SURFACE  = "#161b22"
ACCENT1  = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#f78166"
TEXT     = "#e6edf3"
GRID     = "#30363d"

PALETTE  = [
    "#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657",
    "#79c0ff","#56d364","#ff7b72","#bc8cff","#ffb77a",
    "#a5d6ff","#85e89d","#ffab70","#e6edf3","#8b949e",
    "#30363d","#1f6feb","#238636","#da3633","#6e40c9",
    "#fd8200","#2ea043",
]

def _style():
    """Apply dark figure style."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "axes.grid":         True,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   TEXT,
    })


FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURE_LABELS = {
    "N":           "Nitrogen (N)",
    "P":           "Phosphorus (P)",
    "K":           "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity":    "Humidity (%)",
    "ph":          "Soil pH",
    "rainfall":    "Rainfall (mm)",
}


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap for all numeric features."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    corr = df[FEATURE_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        linecolor=GRID, ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10, "color": TEXT},
    )
    ax.set_title("Feature Correlation Heatmap", pad=15, color=TEXT)
    fig.tight_layout()
    return fig


def plot_feature_distributions(df: pd.DataFrame) -> plt.Figure:
    """KDE distribution of each feature coloured by crop."""
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()

    for idx, col in enumerate(FEATURE_COLS):
        ax = axes[idx]
        for i, crop in enumerate(sorted(df["label"].unique())):
            subset = df[df["label"] == crop][col]
            subset.plot.kde(ax=ax, color=PALETTE[i % len(PALETTE)],
                            alpha=0.6, linewidth=1.5, label=crop)
        ax.set_title(FEATURE_LABELS[col])
        ax.set_xlabel("")
        ax.legend(fontsize=5, ncol=2, framealpha=0.3,
                  loc="upper right", labelcolor=TEXT)

    axes[-1].set_visible(False)  # hide last empty subplot
    fig.suptitle("Feature Distributions by Crop", fontsize=16,
                 fontweight="bold", color=TEXT, y=1.01)
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame, feature: str) -> plt.Figure:
    """Box plot for a single feature across all crops."""
    _style()
    crops_sorted = (
        df.groupby("label")[feature].median()
          .sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    data_per_crop = [df[df["label"] == c][feature].values for c in crops_sorted]
    bp = ax.boxplot(
        data_per_crop, patch_artist=True, notch=False,
        medianprops={"color": ACCENT3, "linewidth": 2},
        whiskerprops={"color": TEXT, "linewidth": 1.2},
        capprops={"color": TEXT, "linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3,
                    "markerfacecolor": ACCENT1, "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(crops_sorted, rotation=40, ha="right", fontsize=9)
    ax.set_title(f"{FEATURE_LABELS[feature]} by Crop Type")
    ax.set_ylabel(FEATURE_LABELS[feature])
    fig.tight_layout()
    return fig


def plot_crop_count(df: pd.DataFrame) -> plt.Figure:
    """Bar chart showing number of records per crop."""
    _style()
    counts = df["label"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    bars = ax.barh(counts.index, counts.values, color=PALETTE[:len(counts)],
                   edgecolor=GRID, linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=9, color=TEXT)
    ax.set_title("Records per Crop Type")
    ax.set_xlabel("Count")
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: np.ndarray,
                            feature_names: list[str]) -> plt.Figure:
    """Horizontal bar chart for Random Forest feature importances."""
    _style()
    sorted_idx = np.argsort(importances)
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i])
              for i in sorted_idx]
    vals   = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    bars = ax.barh(labels, vals * 100, color=PALETTE[:len(labels)],
                   edgecolor=GRID, linewidth=0.5)
    for bar, val in zip(bars, vals * 100):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9, color=TEXT)
    ax.set_title("Random Forest — Feature Importances")
    ax.set_xlabel("Importance (%)")
    ax.set_xlim(0, max(vals * 100) * 1.2)
    fig.tight_layout()
    return fig


def plot_pairplot_top_features(df: pd.DataFrame, top: int = 3) -> plt.Figure:
    """Scatter matrix for the top-N most-correlated features with label."""
    _style()
    # pick top features by variance
    top_feats = (df[FEATURE_COLS].std()
                   .sort_values(ascending=False)
                   .head(top).index.tolist())
    plot_df = df[top_feats + ["label"]].copy()

    crops   = sorted(plot_df["label"].unique())
    color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(crops)}

    fig, axes = plt.subplots(top, top, figsize=(10, 9))
    fig.patch.set_facecolor(BG)

    for i, feat_y in enumerate(top_feats):
        for j, feat_x in enumerate(top_feats):
            ax = axes[i][j]
            if i == j:
                for crop in crops:
                    vals = plot_df[plot_df["label"] == crop][feat_x]
                    vals.plot.kde(ax=ax, color=color_map[crop],
                                  alpha=0.7, linewidth=1.5)
                ax.set_title(FEATURE_LABELS.get(feat_x, feat_x), fontsize=9)
            else:
                for crop in crops:
                    sub = plot_df[plot_df["label"] == crop]
                    ax.scatter(sub[feat_x], sub[feat_y],
                               color=color_map[crop], alpha=0.5, s=10)
            if j == 0:
                ax.set_ylabel(FEATURE_LABELS.get(feat_y, feat_y), fontsize=8)
            if i == top - 1:
                ax.set_xlabel(FEATURE_LABELS.get(feat_x, feat_x), fontsize=8)

    fig.suptitle("Pair Plot — Top 3 High-Variance Features",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.01)
    fig.tight_layout()
    return fig
