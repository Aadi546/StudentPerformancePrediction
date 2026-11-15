"""
Stage-Adaptive Constrained Ensemble Optimizer for CodeWorkout performance prediction.

Optimizes ensemble weights w1 (KNN), w2 (SVR), w3 (XGBoost) for each stage by minimizing
Validation Mean Squared Error (MSE):
    min_{w} MSE(y_val, w1*pred_knn + w2*pred_svr + w3*pred_xgb)
Subject to:
    w1 >= 0, w2 >= 0, w3 >= 0
    w1 + w2 + w3 = 1

Optimization is solved using SciPy's Sequential Least Squares Programming (SLSQP).
Weights are learned separately for every stage and every experimental seed.
Final ensemble predictions are calculated for the untouched Test partition.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def objective_mse(weights: np.ndarray, preds_matrix: np.ndarray, y_true: np.ndarray) -> float:
    """Compute MSE for a weighted combination of predictions."""
    ensemble_pred = np.dot(preds_matrix, weights)
    return float(np.mean((y_true - ensemble_pred) ** 2))


def optimize_weights_slsqp(val_preds_matrix: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """
    Solves the constrained optimization problem via SLSQP.
    Bounds: [0, 1] for each weight.
    Constraint: sum(weights) == 1.
    Initial guess: equal weights [1/3, 1/3, 1/3].
    """
    n_models = val_preds_matrix.shape[1]
    w0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    res = minimize(
        fun=objective_mse,
        x0=w0,
        args=(val_preds_matrix, y_val),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )

    weights = res.x
    # Normalize to ensure exact sum=1 and non-negativity
    weights = np.maximum(0.0, weights)
    weights /= np.sum(weights)
    val_mse = res.fun
    success = res.success

    return weights, success, val_mse


def run_stage_adaptive_ensemble():
    """
    Loads validation predictions for all stages and seeds, optimizes weights,
    computes test ensemble predictions, and saves all learned weights.
    """
    os.makedirs(os.path.join(config.RESULTS_DIR, "weights"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)

    weight_records = []
    test_eval_records = []

    print("Executing Stage-Adaptive Weight Optimization across all 5 stages and 10 seeds...")

    for stage_num in range(1, 6):
        for seed in config.EXPERIMENT_SEEDS:
            pred_file = os.path.join(config.RESULTS_DIR, "predictions", f"stage_{stage_num}_seed_{seed}_preds.json")
            with open(pred_file) as f:
                data = json.load(f)

            y_val = np.array(data['y_val'])
            y_test = np.array(data['y_test'])

            val_knn = np.array(data['val_knn'])
            val_svr = np.array(data['val_svr'])
            val_xgb = np.array(data['val_xgb'])

            test_knn = np.array(data['test_knn'])
            test_svr = np.array(data['test_svr'])
            test_xgb = np.array(data['test_xgb'])

            # Validation prediction matrix: columns = [KNN, SVR, XGB]
            val_matrix = np.column_stack([val_knn, val_svr, val_xgb])
            test_matrix = np.column_stack([test_knn, test_svr, test_xgb])

            # Optimize weights strictly on validation set
            weights, success, val_mse = optimize_weights_slsqp(val_matrix, y_val)
            w_knn, w_svr, w_xgb = weights

            # Apply learned weights to unseen test set
            test_ensemble_preds = np.dot(test_matrix, weights)
            test_mae = mean_absolute_error(y_test, test_ensemble_preds)
            test_rmse = root_mean_squared_error(y_test, test_ensemble_preds)
            test_r2 = r2_score(y_test, test_ensemble_preds)

            # Record weight details
            weight_records.append({
                "stage": stage_num,
                "seed": seed,
                "success": bool(success),
                "val_mse": float(val_mse),
                "val_rmse": float(np.sqrt(val_mse)),
                "w_knn": float(w_knn),
                "w_svr": float(w_svr),
                "w_xgb": float(w_xgb),
            })

            # Record test evaluation details
            test_eval_records.append({
                "stage": stage_num,
                "seed": seed,
                "model": "StageAdaptiveEnsemble",
                "mae": test_mae,
                "rmse": test_rmse,
                "r2": test_r2
            })

            # Append ensemble predictions into the saved json
            data["weights"] = {"w_knn": float(w_knn), "w_svr": float(w_svr), "w_xgb": float(w_xgb)}
            data["test_stage_adaptive"] = test_ensemble_preds.tolist()
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)

    # Save complete weights dataframe
    weights_df = pd.DataFrame(weight_records)
    weights_path = os.path.join(config.RESULTS_DIR, "weights", "stage_adaptive_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    print(f"\nAll learned weights saved to: {weights_path}")

    # Save metrics dataframe
    eval_df = pd.DataFrame(test_eval_records)
    eval_path = os.path.join(config.RESULTS_DIR, "metrics", "stage_adaptive_test_metrics.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Test performance saved to: {eval_path}")

    # Summary table
    print("\n=== LEARNED ENSEMBLE WEIGHTS BY STAGE (MEAN +- STD ACROSS 10 SEEDS) ===")
    summary_w = weights_df.groupby("stage")[["w_knn", "w_svr", "w_xgb"]].agg(["mean", "std"]).round(3)
    print(summary_w)

    print("\n=== STAGE-ADAPTIVE ENSEMBLE TEST PERFORMANCE (MEAN +- STD ACROSS 10 SEEDS) ===")
    summary_p = eval_df.groupby("stage")[["mae", "rmse", "r2"]].agg(["mean", "std"]).round(3)
    print(summary_p)


if __name__ == '__main__':
    run_stage_adaptive_ensemble()
