import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay


def plot_confusion_matrix(cm, labels, out_path: str):
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, out_path: str):
    """Plot and save a ROC curve."""
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_heatmap(grid_df, out_path: str, grid_col: str = "grid_id"):
    """Plot a simple heatmap-like grid from aggregated risk scores."""
    data = grid_df.copy()

    if "_" in str(data[grid_col].iloc[0]):
        parts = data[grid_col].astype(str).str.split("_", expand=True)
        data["tile_x"] = parts[0].astype(int)
        data["tile_y"] = parts[1].astype(int)

    if "tile_x" not in data.columns or "tile_y" not in data.columns:
        raise ValueError("Heatmap requires tile_x and tile_y to plot spatial layout.")

    pivot = data.pivot_table(index="tile_y", columns="tile_x", values="risk_score", fill_value=0)

    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, cmap="YlOrRd")
    plt.xlabel("tile_x")
    plt.ylabel("tile_y")
    plt.title("Congestion Risk Heatmap")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
