import pandas as pd
import joblib
import wandb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Step 1: Init Weights & Biases
wandb.init(
    project="airbnb_model_training",
    name="xgboost_final_manualsplit",
    group="xgboost_experiments",
    config={
        "model": "XGBoost",
        "colsample_bytree": 1,
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 1000,
        "subsample": 0.8,
        "split_note": "manual test set with 1000 positives, 3000 negatives"
    }
)

# Step 2: Load full dataset
df = pd.read_csv("data/processed/train_exp_ready_no_feature_scaling_without_city.csv")

# Step 3: Manually construct test set
df_pos = df[df['high_booking_rate'] == 1]
df_neg = df[df['high_booking_rate'] == 0]

# Sample 2000 positives and 2000 negatives for test set
test_pos = df_pos.sample(n=2000, random_state=42)
test_neg = df_neg.sample(n=2000, random_state=42)
test_df = pd.concat([test_pos, test_neg])

# The rest will be the training set
train_df = df.drop(index=test_df.index)

# Shuffle train and test sets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 4: Prepare features and targets
X_train = train_df.drop(columns=['high_booking_rate'])
y_train = train_df['high_booking_rate']
X_test = test_df.drop(columns=['high_booking_rate'])
y_test = test_df['high_booking_rate']

# pos = y_train.sum()
# neg = len(y_train) - pos
# scale_pos_weight = neg / pos
# print(f"✅ scale_pos_weight = {scale_pos_weight:.2f}")


final_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    colsample_bytree=1,
    learning_rate=0.05,
    max_depth=6,
    n_estimators=1000,
    subsample=0.8,
    scale_pos_weight=4
)

final_model.fit(X_train, y_train)

# Step 6: Save model
model_path = "outputs/models/exp_XGB_golden_sweep_11_3.pkl"
joblib.dump(final_model, model_path)

# Step 7: Evaluate on test set
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print(f"\n✅ AUC on Manually Constructed Test Set: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Log results to wandb
wandb.log({"test_auc": auc})
wandb.save(model_path)
wandb.finish()
