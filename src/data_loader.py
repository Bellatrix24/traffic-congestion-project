import os
from typing import List, Tuple, Optional

import pandas as pd

DEFAULT_FEATURES = [
    "vehicle_density",
    "avg_vehicle_speed",
    "speed_std",
    "lane_occupancy",
    "queue_length",
    "edge_density",
    "optical_flow_mag",
    "shadow_fraction",
    "time_of_day_norm",
    "road_width_norm",
]


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return df


def handle_missing_values(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Fill numeric missing values with median and drop rows missing labels."""
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    df = df.copy()
    df = df.dropna(subset=[label_col])

    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def separate_features_labels(
    df: pd.DataFrame,
    label_col: str = "label",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate features, labels, and spatial metadata if present."""
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]

    if not feature_cols:
        raise ValueError("No feature columns found in the dataset.")

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    meta_cols = [c for c in ["grid_id", "tile_x", "tile_y"] if c in df.columns]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    return X, y, meta
