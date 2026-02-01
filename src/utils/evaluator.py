from sklearn.metrics import roc_auc_score, classification_report

def evaluate_model(y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    print(f"AUC: {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    return auc