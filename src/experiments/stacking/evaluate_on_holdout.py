# src/experiments/stacking/evaluate_on_holdout.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# --- Paths ---
HOLDOUT_PATH = "data/processed/holdout_set.csv"
MODEL_DIR = "outputs/models"
META_MODEL_PATH = os.path.join(MODEL_DIR, "LR_meta_model.pkl")
CONFIG_PATH = "config/stacking_baseline_models.yaml"
TARGET = "high_booking_rate"
N_FOLDS = 5

# --- Load Holdout Set ---
df = pd.read_csv(HOLDOUT_PATH)
y_true = df[TARGET].values
X_holdout = df.drop(columns=[TARGET])
X_holdout = pd.DataFrame(StandardScaler().fit_transform(X_holdout), columns=X_holdout.columns)

# --- Load Base Model Predictions ---
import yaml
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

meta_features = pd.DataFrame(index=df.index)

for model_name in config:
    preds = np.zeros(len(X_holdout))
    for fold in range(N_FOLDS):
        model_path = os.path.join(MODEL_DIR, f"base_model_{model_name}_fold{fold}.pkl")
        model = joblib.load(model_path)
        probas = model.predict_proba(X_holdout)[:, 1]
        preds += probas / N_FOLDS
    meta_features[model_name] = preds

# --- Load Meta-Model ---
meta_model = joblib.load(META_MODEL_PATH)
y_pred = meta_model.predict_proba(meta_features)[:, 1]
auc = roc_auc_score(y_true, y_pred)

print(f"🎯 Final Holdout AUC: {auc:.4f}")