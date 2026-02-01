import pandas as pd
import joblib
import wandb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


# Step 2: Load and split data
df = pd.read_csv("data/processed/train_data_new_v7.csv")


# Prepare features and targets
X_train = df.drop(columns=['high_booking_rate'])
y_train = df['high_booking_rate']

# Step 3: Train final model using best config
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

final_model.fit(X_train, y_train)

# Step 4: Save model
model_path = "outputs/models/XGB_submission_7_balmy-sweep-50.pkl"
joblib.dump(final_model, model_path)

print("Final model trained successfully !!!")

# # Step 5: Evaluate on test set
# y_pred = final_model.predict(X_test)
# y_proba = final_model.predict_proba(X_test)[:, 1]
# auc = roc_auc_score(y_test, y_proba)

# print(f"\n✅ AUC on 5% Test Set: {auc:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))


