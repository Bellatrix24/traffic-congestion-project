from typing import Optional

import pandas as pd


def build_grid_id(df: pd.DataFrame, grid_size: int = 25) -> pd.Series:
    """Build a grid_id from tile_x/tile_y or from row order if missing."""
    if "tile_x" in df.columns and "tile_y" in df.columns:
        return df["tile_x"].astype(str) + "_" + df["tile_y"].astype(str)

    idx = df.index.to_numpy()
    row_bin = (idx // grid_size).astype(int)
    col_bin = (idx % grid_size).astype(int)
    return pd.Series(row_bin.astype(str) + "_" + col_bin.astype(str), index=df.index)


def aggregate_spatial_risk(
    df: pd.DataFrame,
    prob_col: str,
    pred_col: str = "prediction",
    grid_col: Optional[str] = None,
    grid_size: int = 25,
) -> pd.DataFrame:
    """Aggregate predictions to compute congestion risk per grid cell."""
    data = df.copy()

    if grid_col is None:
        if "grid_id" in data.columns:
            grid_col = "grid_id"
        else:
            data["grid_id"] = build_grid_id(data, grid_size=grid_size)
            grid_col = "grid_id"

    grouped = (
        data.groupby(grid_col)
        .agg(
            risk_score=(prob_col, "mean"),
            positive_rate=(pred_col, "mean"),
            samples=(pred_col, "count"),
        )
        .reset_index()
    )

    return grouped
