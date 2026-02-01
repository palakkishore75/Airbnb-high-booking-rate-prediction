# src/experiments/stacking/train_meta_model_top5.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- Paths ---
META_DATA_PATH = "data/processed/meta_dataset_top5.csv"
MODEL_SAVE_PATH = "outputs/models/LR_meta_model_top5.pkl"
LOG_PATH = "outputs/logs/stacking_pipeline_summary_top5.log"

# --- Setup Logging ---
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger()

def log_and_print(msg):
    print(msg)
    log.info(msg)

# --- Load Meta Dataset ---
df = pd.read_csv(META_DATA_PATH)
y = df["high_booking_rate"]
X = df.drop(columns=["high_booking_rate"])

# --- Train Logistic Regression as Meta Model ---
log_and_print("🚀 Training Logistic Regression Meta-Model for Top5")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_preds = np.zeros(len(X))
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    meta_preds[val_idx] = probs
    auc_scores.append(auc)
    log_and_print(f"Fold {fold} AUC: {auc:.4f}")

log_and_print(f"✅ Mean CV AUC: {np.mean(auc_scores):.4f}")

# --- Retrain on Full Data ---
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X, y)
joblib.dump(final_model, MODEL_SAVE_PATH)
log_and_print(f"💾 Meta-model saved to: {MODEL_SAVE_PATH}")
