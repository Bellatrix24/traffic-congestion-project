from typing import Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_model(
    X_train,
    y_train,
    model_type: str = "logistic",
    class_weight: Optional[Dict[int, float]] = None,
):
    """Train a model and return the fitted estimator."""
    if model_type == "logistic":
        model = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            solver="liblinear",
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            class_weight=class_weight,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def get_models() -> Dict[str, str]:
    """Return a simple mapping of model names to type strings."""
    return {
        "Logistic Regression": "logistic",
        "Random Forest": "random_forest",
    }
