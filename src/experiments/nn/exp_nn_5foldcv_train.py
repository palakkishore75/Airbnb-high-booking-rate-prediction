import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# --- Logging Setup ---
log_dir = "outputs/logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "nn_cv_95_5_eval_v2.log")

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()

def log_and_print(msg):
    print(msg)
    log.info(msg)

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 512
PATIENCE = 5
LR = 1e-3
LABEL_SMOOTHING = 0.05
POS_WEIGHT = 4.0  # adjust based on actual class ratio

# --- Label Smoothing ---
def smooth_bce(targets, smoothing=0.05):
    return targets * (1 - smoothing) + 0.5 * smoothing

# --- Load Data ---
df = pd.read_csv("data/processed/train_data_processed_3.csv")
X = df.drop(columns=["high_booking_rate"])
y = df["high_booking_rate"].values

# Impute & Scale
X = SimpleImputer(strategy='median').fit_transform(X)
X = StandardScaler().fit_transform(X)

# Holdout split
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.05, stratify=y, random_state=42
)

# --- Neural Network ---
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)  # no sigmoid here
        )

    def forward(self, x):
        return self.net(x)

# --- Training ---
def train_model(X_train, y_train, X_val, y_val, input_dim, fold=None):
    model = MLP(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ), batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ), batch_size=BATCH_SIZE, shuffle=False)

    best_auc = 0
    best_weights = None
    patience = 0
    train_losses = []
    val_aucs = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = smooth_bce(yb, LABEL_SMOOTHING).to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(yb.numpy())
        auc = roc_auc_score(all_targets, all_preds)
        val_aucs.append(auc)

        log_and_print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_weights = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_weights)

    # Plot
    if fold is not None:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.title(f"Fold {fold} - Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_aucs, label="Val AUC", color="green")
        plt.title(f"Fold {fold} - Val AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return model, best_auc

# --- Cross-validation ---
log_and_print("🚀 Starting 5-Fold Cross-Validation")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    log_and_print(f"\n📦 Fold {fold}")
    model, auc = train_model(X_train[train_idx], y_train[train_idx],
                             X_train[val_idx], y_train[val_idx],
                             input_dim=X_train.shape[1],
                             fold=fold)
    auc_scores.append(auc)
    log_and_print(f"   AUC for fold {fold}: {auc:.4f}")

log_and_print(f"\n✅ Mean CV AUC: {np.mean(auc_scores):.4f}")

# --- Final holdout test ---
final_model, _ = train_model(X_train, y_train, X_holdout, y_holdout, X_train.shape[1])
final_model.eval()

with torch.no_grad():
    Xh = torch.tensor(X_holdout, dtype=torch.float32).to(DEVICE)
    logits = final_model(Xh).squeeze()
    y_pred = torch.sigmoid(logits).cpu().numpy()

holdout_auc = roc_auc_score(y_holdout, y_pred)
log_and_print(f"\n🎯 Final Holdout AUC: {holdout_auc:.4f}")
