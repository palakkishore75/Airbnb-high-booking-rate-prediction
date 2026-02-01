import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from src.model_training.model_defs import get_models_from_config
from src.utils.kfold_runner import run_cv

# Initialize wandb
wandb.init(project="airbnb_model_training", name="exp_kfold_tree_models")

# Load data
file_path = 'data/processed/train_exp_ready_no_feature_scaling.csv'
df = pd.read_csv(file_path)

# Step 1: OOT split - 5% holdout
oot_frac = 0.05
df_remaining, df_oot = train_test_split(df, test_size=oot_frac, stratify=df['high_booking_rate'], random_state=42)

# Step 2: Train-test split from remaining - 80/20
train_df, test_df = train_test_split(df_remaining, test_size=0.2, stratify=df_remaining['high_booking_rate'], random_state=42)

X_train = train_df.drop(columns=['high_booking_rate'])
y_train = train_df['high_booking_rate']

# Load models from YAML config
models = get_models_from_config()
results = {}

# Run CV on each model
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    aucs = run_cv(model, X_train, y_train, n_splits=5, model_name=name)
    results[name] = aucs
