"""
Experimental Evaluation and Statistical Analysis module.

Computes:
1. Mean and standard deviation for MAE, RMSE, R-squared across 10 seeds.
2. Paired statistical significance tests (Paired t-test and Wilcoxon signed-rank test)
   comparing the Proposed Stage-Adaptive Ensemble against:
   - Fixed Stacked Ensemble
   - Equal-Weight Ensemble
   - Best Individual Base Learner
3. Early-stage (Stage 1 & Stage 2) performance audit.
4. Model consistency and variance analysis across seeds.
5. Saves consolidated tables to results/metrics/.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def run_statistical_analysis():
    """
    Performs full statistical evaluation and exports publication-ready tables.
    """
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "all_models_test_metrics.csv")
    df = pd.read_csv(metrics_path)

    # 1. Consolidated Mean & Std Table
    summary = df.groupby(['stage', 'model'])[['mae', 'rmse', 'r2']].agg(['mean', 'std']).reset_index()
    # Flatten columns
    summary.columns = [
        'Stage', 'Model',
        'MAE_Mean', 'MAE_Std',
        'RMSE_Mean', 'RMSE_Std',
        'R2_Mean', 'R2_Std'
    ]
    summary_path = os.path.join(config.RESULTS_DIR, "metrics", "consolidated_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved consolidated metrics summary to: {summary_path}")

    # 2. Paired Statistical Tests against Proposed Method
    models = df['model'].unique()
    test_results = []

    for stage in range(1, 6):
        stage_df = df[df['stage'] == stage]
        adaptive_stage = stage_df[stage_df['model'] == 'StageAdaptiveEnsemble'].sort_values('seed')

        for baseline in ['FixedStackedEnsemble', 'EqualWeightEnsemble', 'XGBoost', 'SVR', 'KNN']:
            base_stage = stage_df[stage_df['model'] == baseline].sort_values('seed')

            # Paired arrays across 10 seeds
            rmse_adapt = adaptive_stage['rmse'].values
            rmse_base = base_stage['rmse'].values

            mae_adapt = adaptive_stage['mae'].values
            mae_base = base_stage['mae'].values

            r2_adapt = adaptive_stage['r2'].values
            r2_base = base_stage['r2'].values

            # Paired t-tests
            t_stat_rmse, p_val_rmse = stats.ttest_rel(rmse_adapt, rmse_base)
            t_stat_mae, p_val_mae = stats.ttest_rel(mae_adapt, mae_base)
            t_stat_r2, p_val_r2 = stats.ttest_rel(r2_adapt, r2_base)

            # Wilcoxon signed rank test
            try:
                w_stat_rmse, w_pval_rmse = stats.wilcoxon(rmse_adapt, rmse_base)
            except Exception:
                w_stat_rmse, w_pval_rmse = np.nan, np.nan

            test_results.append({
                'Stage': stage,
                'Comparison': f"StageAdaptive vs {baseline}",
                'RMSE_Diff (Adapt - Base)': float(np.mean(rmse_adapt - rmse_base)),
                'p_value_RMSE_ttest': float(p_val_rmse),
                'p_value_RMSE_wilcoxon': float(w_pval_rmse),
                'MAE_Diff (Adapt - Base)': float(np.mean(mae_adapt - mae_base)),
                'p_value_MAE_ttest': float(p_val_mae),
                'R2_Diff (Adapt - Base)': float(np.mean(r2_adapt - r2_base)),
                'p_value_R2_ttest': float(p_val_r2)
            })

    tests_df = pd.DataFrame(test_results)
    tests_path = os.path.join(config.RESULTS_DIR, "metrics", "statistical_significance_tests.csv")
    tests_df.to_csv(tests_path, index=False)
    print(f"Saved statistical significance tests to: {tests_path}")

    # 3. Early Stage Audit (Stages 1 and 2)
    early_df = summary[summary['Stage'].isin([1, 2])].copy()
    early_path = os.path.join(config.RESULTS_DIR, "metrics", "early_stage_audit.csv")
    early_df.to_csv(early_path, index=False)
    print(f"Saved early stage audit to: {early_path}")

    # Display key statistical comparisons
    print("\n=== STATISTICAL SIGNIFICANCE SUMMARY: STAGE-ADAPTIVE VS FIXED STACKING ===")
    vs_fixed = tests_df[tests_df['Comparison'] == 'StageAdaptive vs FixedStackedEnsemble']
    print(vs_fixed[['Stage', 'RMSE_Diff (Adapt - Base)', 'p_value_RMSE_ttest', 'R2_Diff (Adapt - Base)', 'p_value_R2_ttest']].to_string(index=False))

    print("\n=== STATISTICAL SIGNIFICANCE SUMMARY: STAGE-ADAPTIVE VS EQUAL-WEIGHT ===")
    vs_equal = tests_df[tests_df['Comparison'] == 'StageAdaptive vs EqualWeightEnsemble']
    print(vs_equal[['Stage', 'RMSE_Diff (Adapt - Base)', 'p_value_RMSE_ttest', 'R2_Diff (Adapt - Base)', 'p_value_R2_ttest']].to_string(index=False))


if __name__ == '__main__':
    run_statistical_analysis()
