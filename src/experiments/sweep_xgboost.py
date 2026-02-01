# File: src/experiments/sweep_xgboost.py

import pandas as pd
import wandb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def sweep_run():
    run = wandb.init(project="airbnb_model_training_phase3", group="xgb_sweep")
    config = run.config

    # Load data
    df = pd.read_csv("data/processed/train_data_new_v8.csv")
    X = df.drop(columns=['high_booking_rate'])
    y = df['high_booking_rate']

    # Split dataset into train (75%) and test (25%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # XGBoost model with full sweep support
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=config.scale_pos_weight,
        gamma=config.gamma,
        min_child_weight=config.min_child_weight
    )

    # Train without early stopping
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics to wandb
    wandb.log({
        "val_auc": auc,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1
    })

    # Optional: log classification report details
    report = classification_report(y_test, y_pred, output_dict=True)
    wandb.log({
        "val_support_0": report['0']['support'],
        "val_support_1": report['1']['support'],
        "val_precision_0": report['0']['precision'],
        "val_precision_1": report['1']['precision'],
        "val_recall_0": report['0']['recall'],
        "val_recall_1": report['1']['recall'],
    })

    run.finish()

if __name__ == "__main__":
    sweep_run()