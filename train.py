import argparse
import os

import pandas as pd

from src.data_loader import load_csv, handle_missing_values, separate_features_labels
from src.preprocessing import compute_class_weights, split_data, scale_features
from src.model import train_model, get_models
from src.evaluation import evaluate_model, explain_accuracy_limitations
from src.spatial_analysis import aggregate_spatial_risk
from src.visualization import plot_confusion_matrix, plot_roc_curve, plot_heatmap


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Traffic Congestion Detection Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save plots")
    parser.add_argument(
        "--grid_size",
        type=int,
        default=25,
        help="Synthetic grid size when spatial columns are missing",
    )
    return parser.parse_args()


def main():
    """Run the end-to-end pipeline."""
    args = parse_args()

    df = load_csv(args.data_path)
    df = handle_missing_values(df, label_col=args.label_col)
    X, y, meta = separate_features_labels(df, label_col=args.label_col)

    X_train, X_val, y_train, y_val = split_data(X, y)
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)

    class_weights = compute_class_weights(y_train)

    results = []
    models = get_models()

    for name, model_type in models.items():
        model = train_model(X_train_scaled, y_train, model_type=model_type, class_weight=class_weights)

        y_pred = model.predict(X_val_scaled)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val_scaled)[:, 1]
        else:
            y_prob = y_pred

        metrics = evaluate_model(y_val, y_pred, y_prob)
        results.append((name, metrics, y_pred, y_prob))

        print(f"\n{name}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-score: {metrics['f1']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print("Confusion Matrix:\n", metrics["confusion_matrix"])

    print("\nWhy accuracy alone is insufficient:")
    print(explain_accuracy_limitations())

    best = sorted(results, key=lambda r: r[1]["f1"], reverse=True)[0]
    best_name, best_metrics, best_pred, best_prob = best

    print(f"\nBest model by F1: {best_name}")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        labels=["normal", "congestion_or_incident"],
        out_path=os.path.join(out_dir, "confusion_matrix.png"),
    )
    plot_roc_curve(y_val, best_prob, out_path=os.path.join(out_dir, "roc_curve.png"))

    val_index = y_val.index
    if meta.empty:
        spatial_df = pd.DataFrame(index=val_index)
    else:
        spatial_df = meta.loc[val_index].copy()

    spatial_df["prediction"] = best_pred
    spatial_df["probability"] = best_prob
    agg = aggregate_spatial_risk(
        spatial_df, prob_col="probability", grid_size=args.grid_size
    )
    agg.to_csv(os.path.join(out_dir, "spatial_risk.csv"), index=False)

    try:
        plot_heatmap(agg, out_path=os.path.join(out_dir, "risk_heatmap.png"))
    except Exception as e:
        print("Heatmap skipped:", e)


if __name__ == "__main__":
    main()
