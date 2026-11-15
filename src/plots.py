"""
Visualization and Result Analysis module.

Generates the 7 required publication-grade plots:
1. MAE comparison across all 5 stages (line plot with error bars).
2. RMSE comparison across all 5 stages (line plot with error bars).
3. R-squared trajectory comparison across all 5 stages.
4. Ensemble weights distribution (w_knn, w_svr, w_xgb) across stages 1-5 (stacked bar chart).
5. Proposed Stage-Adaptive Ensemble vs. Fixed Stacking head-to-head comparison.
6. Early-stage prediction comparison (Stage 1 & Stage 2 zoom).
7. Performance variability / boxplot across the 10 experimental seeds.

All figures saved as high-resolution PNGs (300 DPI) to results/plots/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Set styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

PLOTS_DIR = os.path.join(config.RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_metric_trajectories(df: pd.DataFrame):
    """Plot 1: MAE comparison & Plot 2: RMSE comparison across stages with error bars."""
    models_to_plot = [
        'KNN', 'SVR', 'XGBoost',
        'EqualWeightEnsemble', 'FixedStackedEnsemble', 'StageAdaptiveEnsemble'
    ]
    palette = {
        'KNN': '#95a5a6',
        'SVR': '#3498db',
        'XGBoost': '#e67e22',
        'EqualWeightEnsemble': '#9b59b6',
        'FixedStackedEnsemble': '#e74c3c',
        'StageAdaptiveEnsemble': '#2ecc71'
    }

    # Plot 1: MAE
    plt.figure(figsize=(9, 6))
    for m in models_to_plot:
        sub = df[df['model'] == m].groupby('stage')['mae'].agg(['mean', 'std']).reset_index()
        plt.errorbar(
            sub['stage'], sub['mean'], yerr=sub['std'],
            label=m, color=palette[m], marker='o', linewidth=2.2, capsize=4
        )
    plt.title("Mean Absolute Error (MAE) Across Assignment Stages (Mean ± Std)", pad=12, fontweight='bold')
    plt.xlabel("Cumulative Assignment Stage (1 to 5)")
    plt.ylabel("Test MAE (Grade Points on [0, 100])")
    plt.xticks([1, 2, 3, 4, 5], ['Stage 1\n(A1)', 'Stage 2\n(A1-A2)', 'Stage 3\n(A1-A3)', 'Stage 4\n(A1-A4)', 'Stage 5\n(A1-A5)'])
    plt.legend(frameon=True)
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "01_mae_comparison_across_stages.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # Plot 2: RMSE
    plt.figure(figsize=(9, 6))
    for m in models_to_plot:
        sub = df[df['model'] == m].groupby('stage')['rmse'].agg(['mean', 'std']).reset_index()
        plt.errorbar(
            sub['stage'], sub['mean'], yerr=sub['std'],
            label=m, color=palette[m], marker='s', linewidth=2.2, capsize=4
        )
    plt.title("Root Mean Squared Error (RMSE) Across Assignment Stages (Mean ± Std)", pad=12, fontweight='bold')
    plt.xlabel("Cumulative Assignment Stage (1 to 5)")
    plt.ylabel("Test RMSE (Grade Points on [0, 100])")
    plt.xticks([1, 2, 3, 4, 5], ['Stage 1\n(A1)', 'Stage 2\n(A1-A2)', 'Stage 3\n(A1-A3)', 'Stage 4\n(A1-A4)', 'Stage 5\n(A1-A5)'])
    plt.legend(frameon=True)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "02_rmse_comparison_across_stages.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")


def plot_r2_trajectory(df: pd.DataFrame):
    """Plot 3: R-squared comparison across all 5 stages."""
    palette = {
        'KNN': '#95a5a6',
        'SVR': '#3498db',
        'XGBoost': '#e67e22',
        'EqualWeightEnsemble': '#9b59b6',
        'FixedStackedEnsemble': '#e74c3c',
        'StageAdaptiveEnsemble': '#2ecc71'
    }
    plt.figure(figsize=(9, 6))
    for m in palette.keys():
        sub = df[df['model'] == m].groupby('stage')['r2'].agg(['mean', 'std']).reset_index()
        plt.plot(sub['stage'], sub['mean'], label=m, color=palette[m], marker='D', linewidth=2.2)
        plt.fill_between(sub['stage'], sub['mean'] - sub['std'], sub['mean'] + sub['std'], color=palette[m], alpha=0.1)

    plt.title("Coefficient of Determination (R²) Across Assignment Stages", pad=12, fontweight='bold')
    plt.xlabel("Cumulative Assignment Stage (1 to 5)")
    plt.ylabel("Test R²")
    plt.xticks([1, 2, 3, 4, 5], ['Stage 1\n(A1)', 'Stage 2\n(A1-A2)', 'Stage 3\n(A1-A3)', 'Stage 4\n(A1-A4)', 'Stage 5\n(A1-A5)'])
    plt.legend(frameon=True)
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "03_r2_comparison_across_stages.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")


def plot_ensemble_weights(weights_df: pd.DataFrame):
    """Plot 4: Stacked bar chart of ensemble weight distribution across 5 stages."""
    means = weights_df.groupby('stage')[['w_knn', 'w_svr', 'w_xgb']].mean()

    stages = ['Stage 1\n(A1)', 'Stage 2\n(A1-A2)', 'Stage 3\n(A1-A3)', 'Stage 4\n(A1-A4)', 'Stage 5\n(A1-A5)']
    w_knn = means['w_knn'].values
    w_svr = means['w_svr'].values
    w_xgb = means['w_xgb'].values

    plt.figure(figsize=(8, 6))
    bar_width = 0.55
    x = np.arange(len(stages))

    plt.bar(x, w_knn, width=bar_width, label='KNN Weight (w1)', color='#95a5a6')
    plt.bar(x, w_svr, width=bar_width, bottom=w_knn, label='SVR Weight (w2)', color='#3498db')
    plt.bar(x, w_xgb, width=bar_width, bottom=w_knn + w_svr, label='XGBoost Weight (w3)', color='#e67e22')

    for i in range(len(stages)):
        # Annotate percentages
        if w_knn[i] > 0.05:
            plt.text(i, w_knn[i]/2, f"{w_knn[i]:.1%}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        if w_svr[i] > 0.05:
            plt.text(i, w_knn[i] + w_svr[i]/2, f"{w_svr[i]:.1%}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        if w_xgb[i] > 0.05:
            plt.text(i, w_knn[i] + w_svr[i] + w_xgb[i]/2, f"{w_xgb[i]:.1%}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.title("Stage-Adaptive Ensemble Weight Distribution (SLSQP Optimization)", pad=12, fontweight='bold')
    plt.xlabel("Cumulative Assignment Stage")
    plt.ylabel("Ensemble Weight Contribution (Sum = 1.0)")
    plt.xticks(x, stages)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    p4 = os.path.join(PLOTS_DIR, "04_ensemble_weights_distribution.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")


def plot_proposed_vs_fixed_stacking(df: pd.DataFrame):
    """Plot 5: Head-to-head comparison of Stage-Adaptive vs. Fixed Stacking."""
    sub = df[df['model'].isin(['StageAdaptiveEnsemble', 'FixedStackedEnsemble'])].copy()
    grouped = sub.groupby(['stage', 'model'])['rmse'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(8, 6))
    x = np.arange(1, 6)
    width = 0.35

    adapt = grouped[grouped['model'] == 'StageAdaptiveEnsemble']
    fixed = grouped[grouped['model'] == 'FixedStackedEnsemble']

    plt.bar(x - width/2, adapt['mean'], width, yerr=adapt['std'], label='Proposed Stage-Adaptive', color='#2ecc71', capsize=4)
    plt.bar(x + width/2, fixed['mean'], width, yerr=fixed['std'], label='Fixed Stacking (Linear Reg)', color='#e74c3c', capsize=4)

    plt.title("Head-to-Head: Proposed Stage-Adaptive vs. Fixed Stacking (RMSE)", pad=12, fontweight='bold')
    plt.xlabel("Assignment Stage")
    plt.ylabel("Test RMSE (Lower is Better)")
    plt.xticks(x, ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5'])
    plt.ylim(17, 23)
    plt.legend(frameon=True)
    plt.tight_layout()
    p5 = os.path.join(PLOTS_DIR, "05_proposed_vs_fixed_stacking.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Saved: {p5}")


def plot_early_stage_comparison(df: pd.DataFrame):
    """Plot 6: Early-stage comparison (Stage 1 & Stage 2 zoom)."""
    sub = df[df['stage'].isin([1, 2])].copy()
    order = ['KNN', 'SVR', 'XGBoost', 'EqualWeightEnsemble', 'FixedStackedEnsemble', 'StageAdaptiveEnsemble']
    labels = ['KNN', 'SVR', 'XGBoost', 'Equal-Wt', 'Fixed Stack', 'Stage-Adaptive']

    plt.figure(figsize=(10, 5))
    sns.barplot(data=sub, x='model', y='rmse', hue='stage', order=order, palette=['#3498db', '#e67e22'])
    plt.title("Early-Stage Prediction Performance (Stage 1 vs. Stage 2 RMSE)", pad=12, fontweight='bold')
    plt.xlabel("Model Architecture")
    plt.ylabel("Test RMSE (Lower is Better)")
    plt.xticks(np.arange(len(order)), labels)
    plt.ylim(18, 23)
    plt.legend(title='Assignment Stage', labels=['Stage 1 (A1)', 'Stage 2 (A1-A2)'], frameon=True)
    plt.tight_layout()
    p6 = os.path.join(PLOTS_DIR, "06_early_stage_comparison.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Saved: {p6}")


def plot_performance_variability(df: pd.DataFrame):
    """Plot 7: Performance variability / boxplot across 10 seeds for all models at Stage 5."""
    sub5 = df[df['stage'] == 5].copy()
    order = ['KNN', 'SVR', 'XGBoost', 'EqualWeightEnsemble', 'FixedStackedEnsemble', 'StageAdaptiveEnsemble']
    labels = ['KNN', 'SVR', 'XGBoost', 'Equal-Wt', 'Fixed Stack', 'Stage-Adaptive']

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=sub5, x='model', y='rmse', order=order, palette="Set2")
    sns.stripplot(data=sub5, x='model', y='rmse', order=order, color='black', alpha=0.6, jitter=0.2, size=6)
    plt.title("Model Performance Variability Across 10 Repeated Experiment Seeds (Stage 5)", pad=12, fontweight='bold')
    plt.xlabel("Model Architecture")
    plt.ylabel("Test RMSE (Across 10 Random Seeds)")
    plt.xticks(np.arange(len(order)), labels)
    plt.tight_layout()
    p7 = os.path.join(PLOTS_DIR, "07_performance_variability_across_seeds.png")
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"Saved: {p7}")


def generate_all_plots():
    """Run all 7 plotting routines."""
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "all_models_test_metrics.csv")
    weights_path = os.path.join(config.RESULTS_DIR, "weights", "stage_adaptive_weights.csv")

    df = pd.read_csv(metrics_path)
    w_df = pd.read_csv(weights_path)

    print("Generating all 7 publication-grade visualizations...")
    plot_metric_trajectories(df)
    plot_r2_trajectory(df)
    plot_ensemble_weights(w_df)
    plot_proposed_vs_fixed_stacking(df)
    plot_early_stage_comparison(df)
    plot_performance_variability(df)
    print("\nAll 7 visualization figures successfully generated and saved to results/plots/!")


if __name__ == '__main__':
    generate_all_plots()
