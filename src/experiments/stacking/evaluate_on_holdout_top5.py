# src/experiments/stacking/evaluate_on_holdout_top5.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import yaml

# --- Paths ---
HOLDOUT_PATH = "data/processed/holdout_set.csv"
MODEL_DIR = "outputs/models"
META_MODEL_PATH = os.path.join(MODEL_DIR, "LR_meta_model_top5.pkl")
CONFIG_PATH = "config/stacking_top5_models.yaml"
TARGET = "high_booking_rate"
N_FOLDS = 5

# --- Load Holdout Set ---
df = pd.read_csv(HOLDOUT_PATH)
y_true = df[TARGET].values
X_holdout = df.drop(columns=[TARGET])
X_holdout = pd.DataFrame(StandardScaler().fit_transform(X_holdout), columns=X_holdout.columns)

# --- Load Base Model Config ---
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --- Generate Base Model Predictions ---
meta_features = pd.DataFrame(index=df.index)

for model_name in config:
    preds = np.zeros(len(X_holdout))
    for fold in range(N_FOLDS):
        model_path = os.path.join(MODEL_DIR, f"base_model_{model_name}_fold{fold}_top5.pkl")
        model = joblib.load(model_path)
        probas = model.predict_proba(X_holdout)[:, 1]
        preds += probas / N_FOLDS
    meta_features[model_name] = preds

# --- Load Meta-Model and Predict ---
meta_model = joblib.load(META_MODEL_PATH)
y_pred_proba = meta_model.predict_proba(meta_features)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# --- Evaluation ---
auc = roc_auc_score(y_true, y_pred_proba)
print(f"🎯 Final Holdout AUC (Top5 Stack): {auc:.4f}")

report = classification_report(y_true, y_pred)
print("\n📊 Final Classification Report:\n")
print(report)
