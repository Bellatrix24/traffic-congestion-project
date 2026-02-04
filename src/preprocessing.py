from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def compute_class_weights(y) -> Dict[int, float]:
    """Compute simple inverse-frequency class weights."""
    values, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    weights = {int(v): total / (len(values) * c) for v, c in zip(values, counts)}
    return weights


def split_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split features and labels into train and validation sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_features(
    X_train,
    X_val,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using StandardScaler fitted on training data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler
