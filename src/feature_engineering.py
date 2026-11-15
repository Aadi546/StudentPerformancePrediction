"""
Feature Engineering and Cumulative Stage Builder for CodeWorkout datasets.

Extracts the 7 required programming-pattern features:
1. TimeSpent (cumulative active programming time, idle capped at 600s)
2. Valid (count of syntactically valid submissions)
3. CorrectSub (count of fully correct submissions, score = 1.0)
4. IncorrectSub (count of valid submissions that failed unit tests)
5. CompileError (count of syntax/compilation errors)
6. Scores (mean/max performance score attained across problems)
7. EditDistance (cumulative code modifications across consecutive submissions)

Constructs 5 cumulative assignment stages:
Stage 1: Assignment 439
Stage 2: Assignments 439 + 487
Stage 3: Assignments 439 + 487 + 492
Stage 4: Assignments 439 + 487 + 492 + 494
Stage 5: Assignments 439 + 487 + 492 + 494 + 502
"""

import os
import sys
import difflib
import time
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def compute_event_level_features(activities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes delta time (TimeSpent) and consecutive edit distance (EditDistance)
    for each programming submission event within (Student, Assignment, Problem).
    """
    print("Computing event-level time deltas and edit distances...")
    df = activities_df.sort_values(
        by=['GlobalStudentID', 'AssignmentID', 'ProblemID', 'ServerTimestamp', 'Order']
    ).copy()

    # TimeSpent calculation: difference between consecutive events per problem
    df['PrevTimestamp'] = df.groupby(['GlobalStudentID', 'AssignmentID', 'ProblemID'])['ServerTimestamp'].shift(1)
    time_diff = (df['ServerTimestamp'] - df['PrevTimestamp']).dt.total_seconds().fillna(0.0)
    # Cap idle time at 10 minutes (600 seconds) to avoid inflated times when students take breaks
    df['TimeSpentSec'] = time_diff.clip(lower=0.0, upper=600.0)

    # EditDistance calculation between consecutive code states
    df['PrevCode'] = df.groupby(['GlobalStudentID', 'AssignmentID', 'ProblemID'])['Code'].shift(1)
    
    # Fast vectorized/list calculation for edit distance
    codes_curr = df['Code'].tolist()
    codes_prev = df['PrevCode'].tolist()
    
    edit_distances = []
    print("Calculating code state edit distances...")
    t0 = time.time()
    for prev, curr in zip(codes_prev, codes_curr):
        if prev is None or not isinstance(prev, str) or prev == '':
            edit_distances.append(0.0)
        elif prev == curr:
            edit_distances.append(0.0)
        else:
            # 1 - quick_ratio gives normalized edit distance between 0 and 1
            ratio = difflib.SequenceMatcher(None, prev, curr).quick_ratio()
            edit_distances.append(1.0 - ratio)
            
    t1 = time.time()
    print(f"Edit distances calculated in {t1 - t0:.2f} seconds.")
    df['EditDist'] = edit_distances

    return df


def aggregate_student_features(event_df: pd.DataFrame, target_assignments: list) -> pd.DataFrame:
    """
    Aggregates student programming activity across the specified list of assignments.
    Returns 1 row per student containing the 7 features.
    """
    # Filter to only the cumulative assignments for this stage
    subset = event_df[event_df['AssignmentID'].isin(target_assignments)].copy()

    # Per-problem maximum score achieved by the student
    prob_max_scores = subset.groupby(['GlobalStudentID', 'AssignmentID', 'ProblemID'])['Score'].max().reset_index()
    student_mean_scores = prob_max_scores.groupby('GlobalStudentID')['Score'].mean().rename('Scores')

    # Student-level aggregations
    agg_df = subset.groupby('GlobalStudentID').agg(
        TimeSpent=('TimeSpentSec', 'sum'),
        Valid=('IsValid', 'sum'),
        CorrectSub=('IsCorrect', 'sum'),
        IncorrectSub=('IsIncorrect', 'sum'),
        CompileError=('HasCompileError', 'sum'),
        EditDistance=('EditDist', 'sum'),
    )

    # Combine with mean problem score
    features_df = agg_df.join(student_mean_scores, how='left')
    features_df['Scores'] = features_df['Scores'].fillna(0.0)

    return features_df.reset_index()


def build_all_stages():
    """
    Build and save feature matrices for Stage 1 through Stage 5.
    Aligns each stage with the unified student final grade target.
    """
    print("Loading cleaned activities and student targets...")
    activities = pd.read_parquet(config.CLEANED_ACTIVITIES_PATH)
    targets = pd.read_parquet(config.STUDENT_TARGETS_PATH)

    # Compute event-level TimeSpent and EditDistance
    event_df = compute_event_level_features(activities)

    # Build each cumulative stage
    for stage_num, assignments in config.ASSIGNMENT_STAGES.items():
        print(f"\n--- Building Stage {stage_num} (Assignments: {assignments}) ---")
        stage_feats = aggregate_student_features(event_df, assignments)

        # Merge with targets
        stage_data = targets[['GlobalStudentID', 'Semester', 'FinalGrade']].merge(
            stage_feats, on='GlobalStudentID', how='left'
        )

        # Students with 0 activity up to this stage get 0 for activity counts
        for col in config.FEATURE_NAMES:
            stage_data[col] = stage_data[col].fillna(0.0)

        # Verification checks
        assert len(stage_data) == len(targets), f"Student count mismatch in Stage {stage_num}"
        assert stage_data[config.FEATURE_NAMES].isnull().sum().sum() == 0, f"Nulls found in Stage {stage_num} features"
        assert 'FinalGrade' in stage_data.columns, f"Target missing in Stage {stage_num}"

        stage_file = os.path.join(config.PROCESSED_DATA_DIR, f"stage_{stage_num}_features.parquet")
        stage_data.to_parquet(stage_file, index=False)
        print(f"Stage {stage_num} saved to: {stage_file}")
        print(f"Shape: {stage_data.shape} | Active students with >0 time: {(stage_data['TimeSpent'] > 0).sum()}")

    print("\nAll 5 cumulative assignment stages successfully constructed!")


if __name__ == '__main__':
    build_all_stages()
