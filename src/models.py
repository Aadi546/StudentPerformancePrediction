"""
Base learners module for CodeWorkout performance prediction:
1. K-Nearest Neighbors Regressor (KNN)
2. Support Vector Regressor (SVR)
3. XGBoost Regressor

Features are scaled using StandardScaler fitted SOLELY on training data.
Hyperparameter tuning is conducted using 5-Fold Cross-Validation on training data only.
Validation and test predictions are generated and saved for all stages and seeds.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def tune_knn(X_train_scaled: np.ndarray, y_train: np.ndarray, seed: int) -> KNeighborsRegressor:
    """Tune KNN k and distance metric using 5-fold CV on train data."""
    param_grid = [
        {'n_neighbors': k, 'weights': w, 'metric': m}
        for k in [3, 5, 7, 11, 15, 21]
        for w in ['uniform', 'distance']
        for m in ['euclidean', 'manhattan']
    ]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_score = float('inf')
    best_params = param_grid[0]

    for params in param_grid:
        fold_mses = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            model = KNeighborsRegressor(**params)
            model.fit(X_train_scaled[train_idx], y_train[train_idx])
            preds = model.predict(X_train_scaled[val_idx])
            fold_mses.append(mean_squared_error(y_train[val_idx], preds))
        avg_mse = np.mean(fold_mses)
        if avg_mse < best_score:
            best_score = avg_mse
            best_params = params

    final_model = KNeighborsRegressor(**best_params)
    final_model.fit(X_train_scaled, y_train)
    return final_model


def tune_svr(X_train_scaled: np.ndarray, y_train: np.ndarray, seed: int) -> SVR:
    """Tune SVR C, epsilon, and kernel using 5-fold CV on train data."""
    param_grid = [
        {'C': C, 'epsilon': eps, 'kernel': 'rbf'}
        for C in [0.5, 1.0, 5.0, 10.0, 20.0]
        for eps in [0.1, 0.5, 1.0, 2.0]
    ]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_score = float('inf')
    best_params = param_grid[0]

    for params in param_grid:
        fold_mses = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            model = SVR(**params)
            model.fit(X_train_scaled[train_idx], y_train[train_idx])
            preds = model.predict(X_train_scaled[val_idx])
            fold_mses.append(mean_squared_error(y_train[val_idx], preds))
        avg_mse = np.mean(fold_mses)
        if avg_mse < best_score:
            best_score = avg_mse
            best_params = params

    final_model = SVR(**best_params)
    final_model.fit(X_train_scaled, y_train)
    return final_model


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> XGBRegressor:
    """Tune XGBoost parameters using 5-fold CV on train data."""
    param_grid = [
        {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': 0.8, 'random_state': seed}
        for n in [50, 100]
        for d in [2, 3, 4]
        for lr in [0.03, 0.05, 0.1]
    ]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_score = float('inf')
    best_params = param_grid[0]

    for params in param_grid:
        fold_mses = []
        for train_idx, val_idx in kf.split(X_train):
            model = XGBRegressor(**params)
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict(X_train[val_idx])
            fold_mses.append(mean_squared_error(y_train[val_idx], preds))
        avg_mse = np.mean(fold_mses)
        if avg_mse < best_score:
            best_score = avg_mse
            best_params = params

    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    return final_model


def train_and_predict_all():
    """
    Trains KNN, SVR, and XGBoost across 5 stages and 10 seeds.
    Saves validation and test predictions to results/predictions/.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    preds_dir = os.path.join(config.RESULTS_DIR, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    print("Beginning training of KNN, SVR, and XGBoost across all 5 stages and 10 seeds...")

    for stage_num in range(1, 6):
        stage_file = os.path.join(config.PROCESSED_DATA_DIR, f"stage_{stage_num}_features.parquet")
        stage_df = pd.read_parquet(stage_file)

        for seed in config.EXPERIMENT_SEEDS:
            split_file = os.path.join(config.SPLITS_DIR, f"split_seed_{seed}.json")
            with open(split_file) as f:
                split_dict = json.load(f)

            train_ids = set(split_dict['train_students'])
            val_ids = set(split_dict['val_students'])
            test_ids = set(split_dict['test_students'])

            df_train = stage_df[stage_df['GlobalStudentID'].isin(train_ids)].copy()
            df_val = stage_df[stage_df['GlobalStudentID'].isin(val_ids)].copy()
            df_test = stage_df[stage_df['GlobalStudentID'].isin(test_ids)].copy()

            X_train = df_train[config.FEATURE_NAMES].values
            y_train = df_train['FinalGrade'].values

            X_val = df_val[config.FEATURE_NAMES].values
            y_val = df_val['FinalGrade'].values

            X_test = df_test[config.FEATURE_NAMES].values
            y_test = df_test['FinalGrade'].values

            # Scale strictly on train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # Fit base models
            knn_model = tune_knn(X_train_scaled, y_train, seed)
            val_knn = knn_model.predict(X_val_scaled)
            test_knn = knn_model.predict(X_test_scaled)

            svr_model = tune_svr(X_train_scaled, y_train, seed)
            val_svr = svr_model.predict(X_val_scaled)
            test_svr = svr_model.predict(X_test_scaled)

            xgb_model = tune_xgboost(X_train, y_train, seed)
            val_xgb = xgb_model.predict(X_val)
            test_xgb = xgb_model.predict(X_test)

            # Save predictions dictionary
            record = {
                "stage": stage_num,
                "seed": seed,
                "val_students": df_val['GlobalStudentID'].tolist(),
                "test_students": df_test['GlobalStudentID'].tolist(),
                "y_val": y_val.tolist(),
                "y_test": y_test.tolist(),
                "val_knn": val_knn.tolist(),
                "test_knn": test_knn.tolist(),
                "val_svr": val_svr.tolist(),
                "test_svr": test_svr.tolist(),
                "val_xgb": val_xgb.tolist(),
                "test_xgb": test_xgb.tolist(),
            }

            out_path = os.path.join(preds_dir, f"stage_{stage_num}_seed_{seed}_preds.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

        print(f"Stage {stage_num}/5 complete across all 10 seeds.")

    print("\nAll base models successfully trained and predictions saved to results/predictions/!")


if __name__ == '__main__':
    train_and_predict_all()
