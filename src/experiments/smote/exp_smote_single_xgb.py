# File: src/experiments/stacking/train_xgb_with_smote.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier
import logging

# --- Setup ---
INPUT_PATH = "data/processed/train_data_new_1.csv"
MODEL_SAVE_DIR = "outputs/models"
LOG_DIR = "outputs/logs"
TRAIN_OUTPUT_PATH = "data/processed/train_smote_set.csv"
HOLDOUT_OUTPUT_PATH = "data/processed/holdout_smote_set.csv"
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "xgb_model_smote_v2.pkl")
TARGET = "high_booking_rate"
N_SPLITS = 5
RANDOM_STATE = 42

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(INPUT_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# --- Train/Holdout Split ---
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# Save split datasets
train_df = pd.concat([X_train, y_train], axis=1)
holdout_df = pd.concat([X_holdout, y_holdout], axis=1)
train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
holdout_df.to_csv(HOLDOUT_OUTPUT_PATH, index=False)

# --- Preprocessing ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)

# --- Apply SMOTE ---
sm = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)

# --- Train XGB Model with 5-Fold CV ---
log_path = os.path.join(LOG_DIR, "xgb_smote_training_v2.log")
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger()

def log_and_print(msg):
    print(msg)
    log.info(msg)

log_and_print("🚀 Starting 5-Fold CV with XGBoost")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_smote, y_train_smote), 1):
    X_tr, X_val = X_train_smote[train_idx], X_train_smote[val_idx]
    y_tr, y_val = y_train_smote.iloc[train_idx], y_train_smote.iloc[val_idx]

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        colsample_bytree=0.9,
        learning_rate=0.05,
        max_depth=6,
        n_estimators=1000,
        subsample=0.8,
        scale_pos_weight=1
    )

    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    auc_scores.append(auc)

    log_and_print(f"   AUC for fold {fold}: {auc:.4f}")

log_and_print(f"✅ Mean CV AUC: {np.mean(auc_scores):.4f}")

# --- Train Final Model on Full SMOTEd Data ---
final_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    learning_rate=0.05,
    max_depth=6,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1
)
final_model.fit(X_train_smote, y_train_smote)

# --- Evaluate on Holdout Set ---
holdout_preds = final_model.predict_proba(X_holdout_scaled)[:, 1]
holdout_auc = roc_auc_score(y_holdout, holdout_preds)
log_and_print(f"🎯 Final Holdout AUC: {holdout_auc:.4f}")

# --- Detailed Classification Report ---
holdout_pred_labels = final_model.predict(X_holdout_scaled)
report = classification_report(y_holdout, holdout_pred_labels)
log_and_print("📊 Final Classification Report:\n" + report)

# --- Save Final Model ---
joblib.dump(final_model, FINAL_MODEL_PATH)
log_and_print(f"💾 Final model saved to {FINAL_MODEL_PATH}")