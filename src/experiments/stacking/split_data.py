import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Config ---
INPUT_PATH = "data/processed/train_data_new_1.csv"
TRAIN_OUTPUT_PATH = "data/processed/train_stacking_set.csv"
HOLDOUT_OUTPUT_PATH = "data/processed/holdout_set.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 42

# --- Create Output Folder if Needed ---
os.makedirs("data/processed", exist_ok=True)

# --- Load Full Dataset ---
df = pd.read_csv(INPUT_PATH)

# --- Split into Train (75%) and Holdout (25%) ---
train_df, holdout_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["high_booking_rate"],
    random_state=RANDOM_STATE
)

# --- Save Splits ---
train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
holdout_df.to_csv(HOLDOUT_OUTPUT_PATH, index=False)

print(f"✅ Train set saved to: {TRAIN_OUTPUT_PATH} ({train_df.shape})")
print(f"✅ Holdout set saved to: {HOLDOUT_OUTPUT_PATH} ({holdout_df.shape})")