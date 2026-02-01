# File: src/model_training/generate_submission.py

import pandas as pd
import joblib
import numpy as np

# Load processed test data
test_x = pd.read_csv("data/processed/test_data_x_submission7.csv")

# Load trained model
model = joblib.load("outputs/models/xgb_model_95_5_balmy-sweep-50_v7.pkl")

# Predict probabilities for class 1 (high booking rate = 1)
probs_rate = model.predict_proba(test_x)[:, 1]

# Turn into Series
probs_rate = pd.Series(probs_rate, name="x")

# Step 5: Quick sanity checks
assert not pd.isnull(probs_rate).any(), "There are missing values in the output!"
assert probs_rate.shape[0] == test_x.shape[0], "Mismatch in number of predictions!"

# Step 6: Save as submission file
submission_path = "outputs/submissions/high_booking_rate_group12_submission7.csv"
submission_df = probs_rate.to_frame()
submission_df.to_csv(submission_path, index=False, header=True)

print(f"✅ Submission file created: {submission_path}")
