"""
Master Execution Pipeline for Stage-Adaptive Stacked Ensemble for Early Student Performance Prediction.
Orchestrates Phase 2 through Phase 9 sequentially with full reproducibility.
"""

import os
import sys
import time

def main():
    print("=" * 70)
    print("STAGE-ADAPTIVE STACKED ENSEMBLE FOR EARLY STUDENT PERFORMANCE PREDICTION")
    print("=" * 70)
    t_start = time.time()

    # Step 1: Preprocessing
    print("\n[STEP 1/6] Running Data Preprocessing...")
    from src.preprocessing import run_phase2_preprocessing
    run_phase2_preprocessing()

    # Step 2: Feature Engineering
    print("\n[STEP 2/6] Running Feature Engineering & Cumulative Stage Builder...")
    from src.feature_engineering import build_all_stages
    build_all_stages()

    # Step 3: Student-Level Splitting
    print("\n[STEP 3/6] Running Student-Level Data Splitting (10 Seeds)...")
    from src.splits import generate_student_splits, verify_splits_across_all_stages
    import pandas as pd
    import config
    targets = pd.read_parquet(config.STUDENT_TARGETS_PATH)
    generate_student_splits(targets)
    verify_splits_across_all_stages()

    # Step 4: Base Models Training
    print("\n[STEP 4/6] Training Base Learners (KNN, SVR, XGBoost across 5 Stages & 10 Seeds)...")
    from src.models import train_and_predict_all
    train_and_predict_all()

    # Step 5: Stage-Adaptive Constrained Optimization & Baselines
    print("\n[STEP 5/6] Running Stage-Adaptive SLSQP Optimizer & Ensemble Baselines...")
    from src.weight_optimizer import run_stage_adaptive_ensemble
    from src.ensemble import run_all_baselines
    run_stage_adaptive_ensemble()
    run_all_baselines()

    # Step 6: Evaluation & Visualization
    print("\n[STEP 6/6] Generating Statistical Significance Tests & 7 Publication Plots...")
    from src.evaluation import run_statistical_analysis
    from src.plots import generate_all_plots
    run_statistical_analysis()
    generate_all_plots()

    t_end = time.time()
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE IN {t_end - t_start:.2f} SECONDS!")
    print("=" * 70)

if __name__ == '__main__':
    main()
