import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- Load Data ---
df = pd.read_csv("data/processed/train_data_processed_2.csv")
X = df.drop(columns=["high_booking_rate"])
y = df["high_booking_rate"]
X = X.fillna(-1)

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- LASSO Feature Selection ---
lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
lasso.fit(X_scaled, y)
lasso_importance = pd.Series(np.abs(lasso.coef_[0]), index=X.columns, name="Lasso_Importance")

# --- RFE Feature Selection ---
rfe_selector = RFE(
    estimator=LogisticRegression(solver='liblinear', random_state=42, max_iter=1000),
    n_features_to_select=20
)
rfe_selector = rfe_selector.fit(X_scaled, y)
rfe_ranking = pd.Series(rfe_selector.ranking_, index=X.columns, name="RFE_Rank")
rfe_selected = (rfe_ranking == 1).astype(int)

# --- XGBoost Feature Importance ---
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    colsample_bytree=1,
    learning_rate=0.05,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    scale_pos_weight=4
)
xgb_model.fit(X, y)
xgb_importance = pd.Series(xgb_model.get_booster().get_score(importance_type='gain'))
xgb_importance.index.name = 'Feature'
xgb_importance.name = "XGB_Gain"
xgb_importance = xgb_importance.reindex(X.columns).fillna(0)

# --- Combine Results ---
feature_rank_df = pd.concat([
    lasso_importance,
    rfe_ranking.rename("RFE_Rank"),
    rfe_selected.rename("RFE_Selected"),
    xgb_importance
], axis=1)

feature_rank_df.sort_values(by="Lasso_Importance", ascending=False, inplace=True)

# --- Show Top Features ---
print("\n🔍 Top 30 Features by Lasso Importance:")
print(feature_rank_df)
