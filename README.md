# AI-Based Traffic Congestion & Incident Detection using Aerial Imagery Features

This project builds a clean machine learning pipeline to detect congestion or incidents from pre-extracted aerial imagery features. Each row in the dataset represents a road-segment tile with numeric features, and the label is binary: `normal` vs `congestion_or_incident`.

## Project Structure
```
traffic_congestion_capstone/
│── data/
│── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   ├── spatial_analysis.py
│   └── visualization.py
│── train.py
│── requirements.txt
│── README.md
```
## Features Used

- vehicle_density
- avg_vehicle_speed
- speed_std
- lane_occupancy
- queue_length
- edge_density
- optical_flow_mag
- shadow_fraction
- time_of_day_norm
- road_width_norm

## Model Choice

Two models are trained for comparison:
- Logistic Regression (strong baseline, interpretable)
- Random Forest (handles non-linear patterns)

The best model is picked by F1-score on the validation set.

## Evaluation Metrics

The pipeline reports:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

**Why accuracy is not enough:** traffic incidents are often rare. A model could predict the majority class and still look accurate while missing important congestion events. Precision, recall, and F1 give a more honest signal about detection quality.

## Spatial Aggregation Logic

If the dataset includes `grid_id` or tile coordinates (`tile_x`, `tile_y`), predictions are aggregated per grid cell. The risk score is the mean predicted probability in each cell.  
If spatial columns are missing, the pipeline builds a synthetic grid from row order using fixed-size bins (default `grid_size=25`) to simulate a heatmap layout.

## How To Run

```bash
pip install -r requirements.txt
python train.py --data_path data/your_dataset.csv
```

Outputs:
- `confusion_matrix.png`
- `roc_curve.png`
- `risk_heatmap.png` (if tile coordinates are available)
- `spatial_risk.csv`

## Future Improvements

- Add cross-validation and hyperparameter tuning
- Try gradient boosting models (XGBoost/LightGBM)
- Add calibration for probability outputs
- Integrate time-windowed features for trend detection

