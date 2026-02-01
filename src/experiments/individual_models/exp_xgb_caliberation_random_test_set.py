import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

# Step 1: Load and split data
df = pd.read_csv("data/processed/train_exp_ready_no_feature_scaling_without_city.csv")

# Use 5% for final test set with natural distribution
train_df, test_df = train_test_split(
    df, test_size=0.05, stratify=df['high_booking_rate'], random_state=42
)

# Prepare features and targets
X_train = train_df.drop(columns=['high_booking_rate'])
y_train = train_df['high_booking_rate']
X_test = test_df.drop(columns=['high_booking_rate'])
y_test = test_df['high_booking_rate']

# Step 2: Train base model
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

# Step 3: Calibrate model using isotonic regression
calibrated_model = CalibratedClassifierCV(
    estimator=final_model,
    method='sigmoid',
    cv=3
)
calibrated_model.fit(X_train, y_train)

# Step 4: Evaluate on test set
y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
y_pred_cal = (y_proba_cal >= 0.5).astype(int)

cal_auc = roc_auc_score(y_test, y_proba_cal)
print(f"\n✅ Calibrated AUC on 5% Test Set: {cal_auc:.4f}")
print("\nClassification Report (Calibrated):")
print(classification_report(y_test, y_pred_cal))

# Step 5: Save calibrated model
model_path = "outputs/models/exp1_XGB_golden_sweep_11_calibrated_sigmoid.pkl"
joblib.dump(calibrated_model, model_path)
