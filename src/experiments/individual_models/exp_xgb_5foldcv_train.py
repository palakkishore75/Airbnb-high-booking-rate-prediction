import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# --- CONFIG ---
DATA_PATH = "data/processed/train_data_new_v8.csv"
LOG_PATH = "outputs/logs/xgb_cv_95_5_eval_comfy-sweep-32_v8.log"
MODEL_PATH = "outputs/models/xgb_model_95_5_comfy-sweep-32_v8.pkl"

# --- SETUP LOGGING ---
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger()

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH)
train_df, holdout_df = train_test_split(
    df, test_size=0.05, stratify=df['high_booking_rate'], random_state=42
)

X = train_df.drop(columns=['high_booking_rate'])
y = train_df['high_booking_rate']
X_holdout = holdout_df.drop(columns=['high_booking_rate'])
y_holdout = holdout_df['high_booking_rate']

# --- 5-FOLD CROSS VALIDATION ---
log.info("🚀 Starting 5-Fold CV with XGBoost")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        colsample_bytree=0.8,
        learning_rate=0.01,
        max_depth=9,
        n_estimators=3500,
        subsample=0.8,
        scale_pos_weight=1.5,
        gamma=0,
        min_child_weight=1
    )
    model.fit(X_train, y_train)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_proba)
    auc_scores.append(auc)
    log.info(f"   AUC for fold {fold}: {auc:.4f}")

mean_auc = np.mean(auc_scores)
log.info(f"✅ Mean CV AUC: {mean_auc:.4f}")

# --- FINAL HOLDOUT TEST ---
final_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        colsample_bytree=0.8,
        learning_rate=0.01,
        max_depth=9,
        n_estimators=3500,
        subsample=0.8,
        scale_pos_weight=1.5,
        gamma=0,
        min_child_weight=1
    )
final_model.fit(X, y)
y_holdout_pred = final_model.predict(X_holdout)
y_holdout_proba = final_model.predict_proba(X_holdout)[:, 1]
final_auc = roc_auc_score(y_holdout, y_holdout_proba)

log.info(f"🎯 Final Holdout AUC: {final_auc:.4f}")
log.info("📊 Final Classification Report:\n" + classification_report(y_holdout, y_holdout_pred))

# --- SAVE FINAL MODEL ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(final_model, MODEL_PATH)
log.info(f"💾 Final model saved to {MODEL_PATH}")

print(f"\n✅ Training complete. Log saved to: {LOG_PATH}")
