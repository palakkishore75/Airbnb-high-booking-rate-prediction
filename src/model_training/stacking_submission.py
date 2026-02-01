# File: src/model_training/generate_submission.py

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import yaml

# Load processed test data
X_test = pd.read_csv("data/processed/test_data_x_submission3.csv")

# Load model config
with open("config/stacking_baseline_models.yaml", "r") as f:
    model_config = yaml.safe_load(f)

# Generate base model predictions
N_FOLDS = 5
base_preds = pd.DataFrame(index=X_test.index)

for model_name in model_config:
    preds = np.zeros(len(X_test))
    for fold in range(N_FOLDS):
        model_path = f"outputs/models/base_model_{model_name}_fold{fold}.pkl"
        model = joblib.load(model_path)
        preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
    base_preds[model_name] = preds

# Load meta-model
meta_model = joblib.load("outputs/models/LR_meta_model.pkl")  # meta model is XGB now
final_probs = meta_model.predict_proba(base_preds)[:, 1]

# Turn into Series
probs_rate = pd.Series(final_probs, name="x")

# Sanity checks
assert not pd.isnull(probs_rate).any(), "There are missing values in the output!"
assert probs_rate.shape[0] == X_test.shape[0], "Mismatch in number of predictions!"

# Save submission file
submission_path = "outputs/submissions/high_booking_rate_group12_2.csv"
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission_df = probs_rate.to_frame()
submission_df.to_csv(submission_path, index=False, header=True)

print(f"✅ Submission file created: {submission_path}")
