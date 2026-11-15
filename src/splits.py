"""
Student-level data splitting module for CodeWorkout datasets.

Implements strict student-level partitioning:
- 70% Training
- 15% Validation
- 15% Testing

Generates and saves 10 reproducible splits using fixed random seeds:
[42, 101, 202, 303, 404, 505, 606, 707, 808, 909].

Ensures stratified sampling across semesters (S19 vs F19) and verifies
zero student overlap between train, val, and test splits.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def generate_student_splits(
    targets_df: pd.DataFrame,
    seeds: list = config.EXPERIMENT_SEEDS,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO
) -> dict:
    """
    Generates student-level splits for each seed with semester stratification.
    Saves split metadata as JSON in config.SPLITS_DIR.
    """
    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    all_splits = {}

    unique_students = targets_df[['GlobalStudentID', 'Semester']].drop_duplicates()
    n_total = len(unique_students)
    print(f"Generating student-level splits for {n_total} unique students...")
    print(f"Ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")

    # Relative ratio for test within (val + test)
    val_test_ratio = val_ratio + test_ratio  # 0.30
    test_share_of_val_test = test_ratio / val_test_ratio  # 0.15 / 0.30 = 0.50

    for seed_idx, seed in enumerate(seeds):
        # 1. First split: Train vs (Val + Test) stratified by Semester
        train_df, val_test_df = train_test_split(
            unique_students,
            test_size=val_test_ratio,
            random_state=seed,
            stratify=unique_students['Semester']
        )

        # 2. Second split: Val vs Test stratified by Semester
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=test_share_of_val_test,
            random_state=seed,
            stratify=val_test_df['Semester']
        )

        train_ids = sorted(train_df['GlobalStudentID'].tolist())
        val_ids = sorted(val_df['GlobalStudentID'].tolist())
        test_ids = sorted(test_df['GlobalStudentID'].tolist())

        # Validation assertions
        set_train = set(train_ids)
        set_val = set(val_ids)
        set_test = set(test_ids)

        assert len(set_train.intersection(set_val)) == 0, f"Seed {seed}: Train and Val overlap!"
        assert len(set_train.intersection(set_test)) == 0, f"Seed {seed}: Train and Test overlap!"
        assert len(set_val.intersection(set_test)) == 0, f"Seed {seed}: Val and Test overlap!"
        assert len(set_train) + len(set_val) + len(set_test) == n_total, f"Seed {seed}: Partition count mismatch!"

        split_dict = {
            "experiment_index": seed_idx,
            "seed": seed,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_students": train_ids,
            "val_students": val_ids,
            "test_students": test_ids
        }

        # Save to disk
        split_path = os.path.join(config.SPLITS_DIR, f"split_seed_{seed}.json")
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_dict, f, indent=2)

        all_splits[seed] = split_dict
        print(f"Seed {seed:4d} [Exp {seed_idx+1:2d}/10]: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)} -> Saved {os.path.basename(split_path)}")

    print(f"\nAll {len(seeds)} student-level splits successfully generated and validated!")
    return all_splits


def verify_splits_across_all_stages():
    """
    Checks that every saved split partitions the 5 stage feature files cleanly
    without any missing students or overlap.
    """
    print("\n--- Verifying Splits Against All 5 Cumulative Stages ---")
    for stage_num in range(1, 6):
        stage_file = os.path.join(config.PROCESSED_DATA_DIR, f"stage_{stage_num}_features.parquet")
        stage_df = pd.read_parquet(stage_file)
        stage_student_set = set(stage_df['GlobalStudentID'])

        for seed in config.EXPERIMENT_SEEDS:
            split_path = os.path.join(config.SPLITS_DIR, f"split_seed_{seed}.json")
            with open(split_path, 'r', encoding='utf-8') as f:
                split_dict = json.load(f)

            train_set = set(split_dict['train_students'])
            val_set = set(split_dict['val_students'])
            test_set = set(split_dict['test_students'])

            assert train_set.issubset(stage_student_set), f"Stage {stage_num} missing train students for seed {seed}"
            assert val_set.issubset(stage_student_set), f"Stage {stage_num} missing val students for seed {seed}"
            assert test_set.issubset(stage_student_set), f"Stage {stage_num} missing test students for seed {seed}"

    print("Verification Passed: All 5 stages match the 10 split partitions with 0 data leakage!")


if __name__ == '__main__':
    targets = pd.read_parquet(config.STUDENT_TARGETS_PATH)
    generate_student_splits(targets)
    verify_splits_across_all_stages()
