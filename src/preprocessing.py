"""
Data preprocessing pipeline for CodeWorkout Spring 2019 and Fall 2019 datasets.
Handles cleaning, merging, semester prefixing, target variable unification,
and student activity alignment.
"""

import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import config

def load_and_unify_subjects():
    """
    Load Subject.csv from both semesters, deduplicate, prefix student IDs,
    and unify target final grade to a [0.0, 100.0] scale.
    """
    print("Loading Subject.csv from Spring 2019 and Fall 2019...")
    s19_subj = pd.read_csv(config.S19_SUBJECT).drop_duplicates(subset=['SubjectID'])
    f19_subj = pd.read_csv(config.F19_SUBJECT).drop_duplicates(subset=['SubjectID'])

    # Format IDs with semester prefix to guarantee complete isolation
    s19_subj['Semester'] = 'S19'
    s19_subj['GlobalStudentID'] = 'S19_' + s19_subj['SubjectID'].astype(str)
    # Scale Spring grade from [0, 1] to [0, 100]
    s19_subj['FinalGrade'] = s19_subj['X-Grade'] * 100.0

    f19_subj['Semester'] = 'F19'
    f19_subj['GlobalStudentID'] = 'F19_' + f19_subj['SubjectID'].astype(str)
    # Fall grade is already on [0, 100]
    f19_subj['FinalGrade'] = f19_subj['X-Grade'] * 1.0

    combined_subjects = pd.concat([
        s19_subj[['GlobalStudentID', 'SubjectID', 'Semester', 'FinalGrade']],
        f19_subj[['GlobalStudentID', 'SubjectID', 'Semester', 'FinalGrade']]
    ], ignore_index=True)

    print(f"Total unified subjects: {len(combined_subjects)} (S19: {len(s19_subj)}, F19: {len(f19_subj)})")
    return combined_subjects


def load_and_clean_activities():
    """
    Load MainTable and CodeStates from Spring 2019 and Fall 2019.
    Process runs, compiles, and error counts into aligned programming activity events.
    """
    activities_list = []

    for semester, main_path, cs_path in [
        ('S19', config.S19_MAIN_TABLE, config.S19_CODE_STATES),
        ('F19', config.F19_MAIN_TABLE, config.F19_CODE_STATES)
    ]:
        print(f"\nProcessing {semester} MainTable and CodeStates...")
        df_main = pd.read_csv(main_path, low_memory=False)
        df_cs = pd.read_csv(cs_path, low_memory=False)
        
        # Build code lookup map
        code_map = dict(zip(df_cs['CodeStateID'].astype(str), df_cs['Code'].fillna('')))
        del df_cs

        # Separate by EventType
        # Drop existing empty Compile.* columns from runs so merge has clean single columns
        runs = df_main[df_main['EventType'] == 'Run.Program'].drop(
            columns=['Compile.Result', 'CompileMessageType', 'CompileMessageData'], 
            errors='ignore'
        ).copy()
        compiles = df_main[df_main['EventType'] == 'Compile'].copy()
        errors = df_main[df_main['EventType'] == 'Compile.Error'].copy()

        # Clean IDs for merging
        runs['EventID_clean'] = runs['EventID'].astype(str)
        compiles['EventID_clean'] = compiles['EventID'].astype(str)
        compiles['ParentEventID_clean'] = compiles['ParentEventID'].astype(str).str.replace(r'\.0$', '', regex=True)
        errors['ParentEventID_clean'] = errors['ParentEventID'].astype(str).str.replace(r'\.0$', '', regex=True)

        # Count compile errors per compile event
        err_counts = errors.groupby('ParentEventID_clean').size().rename('CompileErrorCount')

        # Link compile results and error counts to compile events
        compiles = compiles.merge(err_counts, left_on='EventID_clean', right_index=True, how='left')
        compiles['CompileErrorCount'] = compiles['CompileErrorCount'].fillna(0).astype(int)

        # Merge runs with compiles (Run.Program -> Compile child has ParentEventID matching Run's EventID)
        merged_runs = runs.merge(
            compiles[['ParentEventID_clean', 'Compile.Result', 'CompileErrorCount']],
            left_on='EventID_clean',
            right_on='ParentEventID_clean',
            how='left'
        )

        # Build clean activity frame
        n_rows = len(merged_runs)
        act = pd.DataFrame({
            'Semester': [semester] * n_rows,
            'GlobalStudentID': semester + '_' + merged_runs['SubjectID'].astype(str),
            'SubjectID': merged_runs['SubjectID'].astype(str),
            'AssignmentID': merged_runs['AssignmentID'].astype(float).astype(int),
            'ProblemID': merged_runs['ProblemID'].astype(int),
            'Order': merged_runs['Order'].astype(int),
            'ServerTimestamp': pd.to_datetime(merged_runs['ServerTimestamp']),
            'Score': merged_runs['Score'].fillna(0.0).astype(float),
            'CompileResult': merged_runs['Compile.Result'].fillna('Error'),
            'CompileErrorCount': merged_runs['CompileErrorCount'].fillna(0).astype(int),
            'CodeStateID': merged_runs['CodeStateID'].astype(str),
            'Code': merged_runs['CodeStateID'].astype(str).map(code_map).fillna(''),
        })

        # Event classification flags
        act['IsValid'] = (act['CompileResult'] == 'Success').astype(int)
        act['IsCorrect'] = ((act['IsValid'] == 1) & (act['Score'] >= 1.0)).astype(int)
        act['IsIncorrect'] = ((act['IsValid'] == 1) & (act['Score'] < 1.0)).astype(int)
        act['HasCompileError'] = ((act['CompileResult'] == 'Error') | (act['CompileErrorCount'] > 0)).astype(int)

        print(f"Loaded {len(act)} execution records for {semester}.")
        activities_list.append(act)

    all_activities = pd.concat(activities_list, ignore_index=True)
    # Sort chronologically by student, problem, and timestamp/order
    all_activities.sort_values(by=['GlobalStudentID', 'AssignmentID', 'ProblemID', 'ServerTimestamp', 'Order'], inplace=True)
    all_activities.reset_index(drop=True, inplace=True)
    
    return all_activities


def run_phase2_preprocessing():
    """
    Execute full Phase 2 preprocessing and save verified outputs.
    """
    subjects_df = load_and_unify_subjects()
    activities_df = load_and_clean_activities()

    # Filter activities to only students present in Subject.csv
    valid_student_ids = set(subjects_df['GlobalStudentID'])
    activities_df = activities_df[activities_df['GlobalStudentID'].isin(valid_student_ids)].copy()

    # Keep students with positive activity
    students_with_activity = set(activities_df['GlobalStudentID'].unique())
    print(f"\nStudents in Subject: {len(valid_student_ids)}")
    print(f"Students with valid activity: {len(students_with_activity)}")

    # Save to parquet for fast and faithful reloading
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    subjects_df.to_parquet(config.STUDENT_TARGETS_PATH, index=False)
    activities_df.to_parquet(config.CLEANED_ACTIVITIES_PATH, index=False)

    print(f"\nSaved student targets to: {config.STUDENT_TARGETS_PATH}")
    print(f"Saved cleaned activities to: {config.CLEANED_ACTIVITIES_PATH}")
    print(f"Total cleaned activity events: {len(activities_df):,}")

    return subjects_df, activities_df


if __name__ == '__main__':
    run_phase2_preprocessing()
