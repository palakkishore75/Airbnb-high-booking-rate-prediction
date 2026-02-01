from sklearn.model_selection import StratifiedKFold
from src.utils.evaluator import evaluate_model
import numpy as np
import os
import joblib
import wandb

def run_cv(model, X, y, n_splits=5, model_name="Model"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []

    os.makedirs('outputs/models', exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        auc = evaluate_model(y_val, y_pred, y_proba)
        auc_scores.append(auc)

        # Save model for this fold
        model_path = f"outputs/models/{model_name}_fold{fold+1}.pkl"
        joblib.dump(model, model_path)

    # Final summary logging
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    wandb.log({
        f"{model_name}_AUCs": auc_scores,
        f"{model_name}_mean_AUC": mean_auc,
        f"{model_name}_std_AUC": std_auc,
        "model_summary": wandb.Table(
            data=[[model_name, mean_auc, std_auc]],
            columns=["Model", "Mean AUC", "Std AUC"]
        )
    })

    print(f"\n{model_name} Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    return auc_scores
