import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from pytorch_tabular import TabularModel
from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig

import torch
import torch.nn as nn

pos_weight = torch.tensor([4.0])  # Adjust based on your 80:20 class ratio
custom_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# --- Logging Setup ---
log_dir = "outputs/logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "fttransformer_cv_eval.log")
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()

def log_and_print(msg):
    print(msg)
    log.info(msg)

# # --- Top 30 Selected Features ---
# selected_features = [
#     'host_response_time_encoded', 'host_acceptance_rate_binned', 'min_night_bin', 
#     'host_acceptance_rate', 'booking_pressure', 'price_per_guest', 'cleaning_fee', 
#     'availability_30', 'price_log', 'availability_90', 'guests_included_encoded', 
#     'room_type_encoded', 'cancellation_policy_encoded', 'availability_rate_30', 
#     'lat_long_product', 'listing_age_days', 'host_account_age', 'rooms_per_guest', 
#     'extra_people_log', 'availability_60', 'bedrooms', 'beds', 'accommodates', 
#     'minimum_nights', 'has_cleaning_fee', 'availability_rate_365', 'price_per_bedroom', 
#     'availability_365', 'price_x_guests', 'property_type_encoded'
# ]

# --- Load and Prepare Data ---
df = pd.read_csv("data/processed/train_data_processed_2.csv")
# df = df[selected_features + ["high_booking_rate"]]

# Impute and scale
X = df.drop(columns=["high_booking_rate"])
y = df["high_booking_rate"]

X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
df_processed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

# --- 5-Fold Stratified CV ---
log_and_print("🚀 Starting FT-Transformer 5-Fold CV")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    log_and_print(f"\n📦 Fold {fold}")

    train_df = df_processed.iloc[train_idx].copy()
    val_df = df_processed.iloc[val_idx].copy()

    data_config = DataConfig(
        target="high_booking_rate",
        continuous_cols=train_df.columns,
        categorical_cols=[],
    )

    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        metrics=["auc"],
        seed=42,
        batch_norm_continuous_input=True,
        attention_dropout=0.1,
        num_attention_blocks=2,
        attention_heads=4,
        ff_hidden_multiplier=2,
        dropout=0.1
    )

    trainer_config = TrainerConfig(
        max_epochs=50,
        batch_size=512,
        early_stopping=True,
        early_stopping_patience=5,
        checkpoints=None,
        progress_bar=False,
        loss=custom_loss
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
    )

    tabular_model.fit(train=train_df, validation=val_df)

    preds = tabular_model.predict(val_df)
    auc = roc_auc_score(val_df["high_booking_rate"], preds["prediction"])
    auc_scores.append(auc)

    log_and_print(f"   AUC for fold {fold}: {auc:.4f}")

log_and_print(f"\n✅ Mean CV AUC: {np.mean(auc_scores):.4f}")
