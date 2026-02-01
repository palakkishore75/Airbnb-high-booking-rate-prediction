# File: 3_model_blend_train.py

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# --- CONFIG ---
DATA_PATH = "data/processed/train_data_new_v4.csv"
LOG_PATH = "outputs/logs/blend_model_95_5_eval.log"
MODEL_DIR = "outputs/models/blend_models/"

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

# --- INITIALIZE MODELS ---
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    colsample_bytree=0.9,
    learning_rate=0.01,
    max_depth=9,
    n_estimators=3000,
    subsample=0.8,
    scale_pos_weight=3,
    gamma=1,
    min_child_weight=1
)

lgb_model = LGBMClassifier(
    random_state=42,
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=9,
    subsample=0.8,
    colsample_bytree=0.9,
    scale_pos_weight=3
)

cat_model = CatBoostClassifier(
    random_state=42,
    iterations=2000,
    learning_rate=0.01,
    depth=9,
    verbose=0,
    scale_pos_weight=3
)

# --- TRAIN ALL 3 MODELS ON FULL TRAIN SET ---
xgb_model.fit(X, y)
lgb_model.fit(X, y)
cat_model.fit(X, y)

# --- PREDICT ON HOLDOUT ---
xgb_preds = xgb_model.predict_proba(X_holdout)[:, 1]
lgb_preds = lgb_model.predict_proba(X_holdout)[:, 1]
cat_preds = cat_model.predict_proba(X_holdout)[:, 1]

# --- BLEND PREDICTIONS (WEIGHTED AVERAGE) ---
final_preds = (0.4 * xgb_preds) + (0.3 * lgb_preds) + (0.3 * cat_preds)
final_pred_labels = (final_preds > 0.5).astype(int)

# --- EVALUATE ---
final_auc = roc_auc_score(y_holdout, final_preds)
log.info(f"🎯 Final Holdout AUC (Blended): {final_auc:.4f}")
log.info("📊 Final Classification Report (Blended):\n" + classification_report(y_holdout, final_pred_labels))

# --- SAVE MODELS ---
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model_blend.pkl"))
joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgb_model_blend.pkl"))
joblib.dump(cat_model, os.path.join(MODEL_DIR, "cat_model_blend.pkl"))
log.info(f"💾 All models saved to {MODEL_DIR}")

print(f"\n✅ 3-model blend complete. Log saved to: {LOG_PATH}")