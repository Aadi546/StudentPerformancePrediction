"""
Ensemble module for CodeWorkout performance prediction:
1. Equal-Weight Ensemble: (KNN + SVR + XGBoost) / 3
2. Fixed Stacked Ensemble: Meta-learner (Linear Regression) trained strictly
   on Out-of-Fold (OOF) cross-validation predictions of the training partition.

Evaluates all baseline models against the Proposed Stage-Adaptive Ensemble
across all 5 cumulative stages and 10 random seeds.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def generate_oof_predictions(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    """
    Generates Out-of-Fold (OOF) predictions on the training data using 5-fold CV.
    Returns an (N_train, 3) matrix containing [OOF_KNN, OOF_SVR, OOF_XGB].
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_knn = np.zeros(len(y_train))
    oof_svr = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))

    for fold_train_idx, fold_val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[fold_train_idx], y_train[fold_train_idx]
        X_va, y_va = X_train[fold_val_idx], y_train[fold_val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # KNN
        knn = KNeighborsRegressor(n_neighbors=11, weights='distance', metric='manhattan')
        knn.fit(X_tr_scaled, y_tr)
        oof_knn[fold_val_idx] = knn.predict(X_va_scaled)

        # SVR
        svr = SVR(kernel='rbf', C=5.0, epsilon=0.5)
        svr.fit(X_tr_scaled, y_tr)
        oof_svr[fold_val_idx] = svr.predict(X_va_scaled)

        # XGBoost
        xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
        xgb.fit(X_tr, y_tr)
        oof_xgb[fold_val_idx] = xgb.predict(X_va)

    return np.column_stack([oof_knn, oof_svr, oof_xgb])


def run_all_baselines():
    """
    Evaluates Equal-Weight Ensemble and Fixed Stacked Ensemble across all stages and seeds.
    Combines with base learners and proposed stage-adaptive ensemble into master results table.
    """
    records = []

    print("Running Equal-Weight Ensemble and Fixed Stacking (OOF Meta-Learner)...")

    for stage_num in range(1, 6):
        stage_file = os.path.join(config.PROCESSED_DATA_DIR, f"stage_{stage_num}_features.parquet")
        stage_df = pd.read_parquet(stage_file)

        for seed in config.EXPERIMENT_SEEDS:
            pred_file = os.path.join(config.RESULTS_DIR, "predictions", f"stage_{stage_num}_seed_{seed}_preds.json")
            with open(pred_file) as f:
                data = json.load(f)

            split_file = os.path.join(config.SPLITS_DIR, f"split_seed_{seed}.json")
            with open(split_file) as f:
                split_dict = json.load(f)

            train_ids = set(split_dict['train_students'])
            df_train = stage_df[stage_df['GlobalStudentID'].isin(train_ids)].copy()

            X_train = df_train[config.FEATURE_NAMES].values
            y_train = df_train['FinalGrade'].values

            y_test = np.array(data['y_test'])
            test_knn = np.array(data['test_knn'])
            test_svr = np.array(data['test_svr'])
            test_xgb = np.array(data['test_xgb'])
            test_matrix = np.column_stack([test_knn, test_svr, test_xgb])

            # 1. Individual Base Models
            for name, preds in [('KNN', test_knn), ('SVR', test_svr), ('XGBoost', test_xgb)]:
                records.append({
                    'stage': stage_num,
                    'seed': seed,
                    'model': name,
                    'mae': mean_absolute_error(y_test, preds),
                    'rmse': root_mean_squared_error(y_test, preds),
                    'r2': r2_score(y_test, preds)
                })

            # 2. Baseline 4: Equal-Weight Ensemble
            test_equal = (test_knn + test_svr + test_xgb) / 3.0
            records.append({
                'stage': stage_num,
                'seed': seed,
                'model': 'EqualWeightEnsemble',
                'mae': mean_absolute_error(y_test, test_equal),
                'rmse': root_mean_squared_error(y_test, test_equal),
                'r2': r2_score(y_test, test_equal)
            })

            # 3. Baseline 5: Fixed Stacking (Linear Regression Meta-Learner on OOF predictions)
            oof_train_preds = generate_oof_predictions(X_train, y_train, seed)
            meta_learner = LinearRegression()
            meta_learner.fit(oof_train_preds, y_train)

            test_stacking = meta_learner.predict(test_matrix)
            records.append({
                'stage': stage_num,
                'seed': seed,
                'model': 'FixedStackedEnsemble',
                'mae': mean_absolute_error(y_test, test_stacking),
                'rmse': root_mean_squared_error(y_test, test_stacking),
                'r2': r2_score(y_test, test_stacking)
            })

            # 4. Proposed Method: Stage-Adaptive Ensemble
            test_adaptive = np.array(data['test_stage_adaptive'])
            records.append({
                'stage': stage_num,
                'seed': seed,
                'model': 'StageAdaptiveEnsemble',
                'mae': mean_absolute_error(y_test, test_adaptive),
                'rmse': root_mean_squared_error(y_test, test_adaptive),
                'r2': r2_score(y_test, test_adaptive)
            })

            # Update prediction file with baselines
            data['test_equal_weight'] = test_equal.tolist()
            data['test_fixed_stacking'] = test_stacking.tolist()
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)

        print(f"Stage {stage_num}/5 complete across all 6 models and 10 seeds.")

    # Save master evaluation dataframe
    all_metrics_df = pd.DataFrame(records)
    out_csv = os.path.join(config.RESULTS_DIR, "metrics", "all_models_test_metrics.csv")
    all_metrics_df.to_csv(out_csv, index=False)
    print(f"\nMaster experimental metrics saved to: {out_csv}")

    # Display full comparison table
    summary = all_metrics_df.groupby(['stage', 'model'])[['mae', 'rmse', 'r2']].agg(['mean', 'std']).round(3)
    print("\n=== COMPREHENSIVE 6-MODEL COMPARISON (MEAN +- STD ACROSS 10 SEEDS) ===")
    print(summary)


if __name__ == '__main__':
    run_all_baselines()
