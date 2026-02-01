import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# Load processed data
df = pd.read_csv("data/processed/train_exp_ready_no_feature_scaling.csv")

# Step 1: Remove OOT (5%) just like in training
df_remaining, df_oot = train_test_split(df, test_size=0.05, stratify=df['high_booking_rate'], random_state=42)

# Step 2: Load the 20% test set
train_df, test_df = train_test_split(df_remaining, test_size=0.2, stratify=df_remaining['high_booking_rate'], random_state=42)

X_test = test_df.drop(columns=['high_booking_rate'])
y_test = test_df['high_booking_rate']

# Step 3: Load the trained XGBoost model (e.g., from fold 1)
model = joblib.load("outputs/models/XGBoost_fold1.pkl")

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print(f"AUC on 20% test set: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
