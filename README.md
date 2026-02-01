# Airbnb High Booking Rate Prediction 🏡📈

This repository contains an end-to-end machine learning pipeline for predicting **high booking rate Airbnb listings** (binary classification).  
The final solution achieved a **0.916 AUC on the hidden test set**, securing **2nd place** in the competition.

The project emphasizes clean experimentation, modular training pipelines, and reproducibility, with a strong focus on tree-based models and ensemble techniques.

---

## 🏆 Results

- **Final Hidden Test AUC:** **0.916**
- **Leaderboard Position:** **2nd**
- **Primary Metric:** ROC-AUC

---

## 🧠 High-Level Approach

- Feature engineering and feature selection experiments
- K-fold cross-validation for robust model evaluation
- Extensive **XGBoost tuning** (manual + W&B sweeps)
- Experiments with imbalance handling, neural networks, and stacking
- Final retraining and submission generation from selected configurations

This README intentionally avoids documenting every experiment in detail; instead, it focuses on helping a new reader understand **where things live** and **how the pipeline fits together**.

---

## 📁 Repository Structure

```
.
├── config/                          # Centralized configuration files
│   ├── model_config.yaml            # Core model & training configuration
│   ├── sweep_config.yaml            # W&B hyperparameter sweep configuration
│   ├── stacking_baseline_models.yaml# Baseline stacking model definitions
│   ├── stacking_top5_models.yaml    # Top-5 stacking configuration
│   └── stacking_xgb_only.yaml       # XGBoost-only stacking setup
│
├── docs/                            # Project documentation / reports
├── notebooks/                       # EDA and exploratory notebooks
├── outputs/                         # Saved models, metrics, predictions, submissions
│
├── src/
│   ├── experiments/                 # Experimental & exploratory pipelines
│   │   ├── individual_models/       # Single-model baselines
│   │   ├── nn/                      # Neural network experiments
│   │   ├── smote/                   # Imbalance handling (SMOTE) experiments
│   │   ├── stacking/                # Stacking experiments and prototypes
│   │   └── sweep_xgboost.py         # XGBoost hyperparameter sweeps (W&B)
│   │
│   ├── feature_engineering/
│   │   └── feature_selection_lasso_rfe.py
│   │                                  # Feature selection via LASSO and RFE
│
│   ├── model_training/              # Main training and inference pipeline
│   │   ├── model_defs.py             # Centralized model definitions
│   │   ├── run_kfold_training.py     # K-fold CV training entrypoint
│   │   ├── retrain_final_xgboost.py  # Retrain best model on full training data
│   │   ├── evaluate_test_set.py      # Evaluation on held-out data (if applicable)
│   │   ├── generate_submission.py    # Generate competition submission
│   │   └── stacking_submission.py   # Submission pipeline for stacking models
│   │
│   └── utils/                       # Shared utilities (logging, IO, metrics)
│
├── wandb/                           # Weights & Biases run artifacts (local)
├── LICENSE
└── README.md

````

---

## ⚙️ Configuration Files (`config/`)

All experiments and pipelines are driven by YAML configs:

- **`model_config.yaml`**  
  Core training configuration (model parameters, CV setup, paths, seeds).

- **`sweep_config.yaml`**  
  Weights & Biases sweep configuration for XGBoost hyperparameter tuning.

- **`stacking_baseline_models.yaml`**  
  Defines baseline models used in stacking experiments.

- **`stacking_top5_models.yaml`**  
  Stacking configuration using the top-performing individual models.

- **`stacking_xgb_only.yaml`**  
  Stacking setup restricted to XGBoost variants only.

---

## 🛠️ Tech Stack

**Core Libraries**
- Python
- Pandas, NumPy
- scikit-learn
- XGBoost

**Experiment Tracking**
- **Weights & Biases (wandb)**  
  - Hyperparameter sweeps  
  - Metric tracking  
  - Run comparison and reproducibility  

**Other Tools**
- Matplotlib / Seaborn (EDA & diagnostics)
- YAML-based configuration management
- Modular logging and utilities

---

## ▶️ Typical Workflow

### 1️⃣ Cross-Validation Training
Runs K-fold CV and logs metrics.

```bash
python src/model_training/run_kfold_training.py
````

### 2️⃣ Retrain Final Model

Retrains the best-performing configuration on full training data.

```bash
python src/model_training/retrain_final_xgboost.py
```

### 3️⃣ Generate Submission

Creates the final submission file.

```bash
python src/model_training/generate_submission.py
```

### (Optional) Stacking Submission

```bash
python src/model_training/stacking_submission.py
```

---

## 🧪 Notes on Experiments

The `src/experiments/` directory contains exploratory work and alternative modeling strategies.
The **primary, production-style pipeline** lives in `src/model_training/`.

---

## 📌 License

See `LICENSE`.


