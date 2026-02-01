## `run_kfold_training.py` Workflow
 - Load train_exp_ready_no_feature_scaling.csv
 - Drop target column → X, extract high_booking_rate → y
 - get_models() → from model_defs.py
    * For each model:
        - run_cv(model, X, y) → from kfold_runner.py
        - For each fold:
            - Train model
            - Predict y_pred, y_proba
            - evaluate_model(y_true, y_pred, y_proba) → from evaluator.py
            - Print AUC + classification report

 - Print mean AUC for each model