# src/experiments/stacking/train_base_models_top5.py

import os
import yaml
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- Setup ---
CONFIG_PATH = "config/stacking_top5_models.yaml"
DATA_PATH = "data/processed/train_stacking_set.csv"
MODEL_DIR = "outputs/models"
LOG_DIR = "outputs/logs"
META_DATASET_PATH = "data/processed/meta_dataset_top5.csv"
TARGET = "high_booking_rate"
N_SPLITS = 5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET].values
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# --- Load Model Config ---
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --- Store OOF predictions ---
oof_preds = pd.DataFrame(index=df.index)
oof_preds[TARGET] = y

# --- Train Models ---
for model_name, model_cfg in config.items():
    log_file = os.path.join(LOG_DIR, f"stacking_model_{model_name}_top5_v1.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s | %(message)s')
    logging.info(f"\n🔧 Training model: {model_name}")

    model_type = model_cfg["model_type"]
    model_class = None

    if model_type == "xgb":
        model_class = XGBClassifier
    elif model_type == "lgb":
        model_class = LGBMClassifier
    elif model_type == "catboost":
        model_class = CatBoostClassifier
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = model_class(**model_cfg.get("params", {}))

        model.fit(X_train, y_train)
        probas = model.predict_proba(X_val)[:, 1]

        oof_pred[val_idx] = probas

        model_path = os.path.join(MODEL_DIR, f"base_model_{model_name}_fold{fold}_top5.pkl")
        joblib.dump(model, model_path)
        logging.info(f"✅ Fold {fold} AUC: {roc_auc_score(y_val, probas):.4f}")

    oof_preds[model_name] = oof_pred
    logging.info(f"✅ Mean OOF AUC for {model_name}: {roc_auc_score(y, oof_pred):.4f}")

# --- Save Meta Dataset ---
oof_preds.to_csv(META_DATASET_PATH, index=False)
print(f"📦 Meta dataset saved to: {META_DATASET_PATH}")
