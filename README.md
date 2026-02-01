
# Airbnb High Booking Rate Prediction 🏡📈

This repository contains the complete pipeline for a machine learning competition project focused on predicting **high booking rate listings on Airbnb**.  
The final model achieved an **AUC of 0.916 on the hidden test set**, securing **2nd place** on the competition leaderboard.

The project emphasizes clean feature engineering, modular experimentation, robust model evaluation, and reproducibility.

---

## 📌 Problem Overview

The goal of this project is to build a binary classification model that predicts whether an Airbnb listing will have a **high booking rate**, based on listing attributes, host behavior, pricing signals, and historical patterns.

This is a real-world style ML problem involving:
- Tabular data
- Class imbalance
- Feature-rich inputs
- Careful validation and leakage prevention

---

## 🧠 Modeling Approach (High-Level)

- Extensive **feature engineering** driven by domain intuition
- Strong baseline models followed by **XGBoost-focused experimentation**
- **K-fold cross-validation** with AUC as the primary metric
- Final model selection based on CV stability and leaderboard performance
- Reproducible experiment tracking and logging

> Note: This README intentionally avoids detailing every experiment to keep the repository approachable. See code and configs for deeper dives.

---

## 📂 Repository Structure

```

├── config/
│   ├── model_config.yaml
│   └── feature_config.yaml
│
├── data/
│   ├── raw/
│   │   └── original competition datasets
│   ├── processed/
│   │   ├── processed_train_x.csv
│   │   ├── processed_train_y.csv
│   │   └── processed_test_x.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaning.py
│   │   ├── feature_engineering.py
│   │   └── feature_validation.py
│   │
│   ├── experiments/
│   │   ├── train_xgboost.py
│   │   ├── cross_validation.py
│   │   └── evaluate_models.py
│   │
│   ├── inference/
│   │   └── generate_predictions.py
│   │
│   └── utils/
│       ├── logging_utils.py
│       ├── metrics.py
│       └── io_utils.py
│
├── outputs/
│   ├── models/
│   │   └── trained model artifacts
│   ├── logs/
│   │   └── training and evaluation logs
│   └── submissions/
│       └── final competition submission files
│
├── notebooks/
│   └── exploratory analysis and sanity checks
│
├── requirements.txt
├── README.md
└── run_pipeline.py

````

---

## 🛠️ Technologies & Tools Used

**Core Stack**
- Python
- Pandas, NumPy
- scikit-learn
- XGBoost

**Experiment Tracking & Monitoring**
- Weights & Biases (W&B) for:
  - Experiment tracking
  - Metric comparison
  - Hyperparameter logging
  - Model versioning

**Other Tools**
- Matplotlib / Seaborn (EDA & diagnostics)
- YAML-based configuration management
- Modular logging with Python `logging`

---

## ▶️ How to Run the Project

1. **Install dependencies**
```bash
pip install -r requirements.txt
````

2. **Run the full pipeline**

```bash
python run_pipeline.py
```

3. **Generate test predictions**

```bash
python src/inference/generate_predictions.py
```

---

## 📊 Evaluation Metric

* **Primary Metric:** ROC-AUC
* **Validation Strategy:** K-Fold Cross Validation
* **Final Result:**

  * **Hidden Test AUC:** `0.916`
  * **Leaderboard Position:** `2nd Place`

---

## 🧩 Key Design Principles

* Modular and reusable code structure
* Strict separation of data processing, modeling, and inference
* Configuration-driven experimentation
* Emphasis on reproducibility and traceability

---

## 🚀 Future Improvements

* Model stacking / ensembling
* Feature group ablations
* SHAP-based interpretability
* Calibration analysis (Brier score, reliability plots)

---

## 📬 Contact

If you have questions about the project structure or want to extend this work, feel free to reach out or open an issue.

---

**Built with care, iteration, and far too many AUC plots.**


---
