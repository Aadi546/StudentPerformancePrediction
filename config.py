import os

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Semester data paths
S19_RAW_DIR = os.path.join(RAW_DATA_DIR, "spring_2019")
F19_RAW_DIR = os.path.join(RAW_DATA_DIR, "fall_2019")

S19_MAIN_TABLE = os.path.join(S19_RAW_DIR, "Data", "MainTable.csv")
S19_SUBJECT = os.path.join(S19_RAW_DIR, "Data", "LinkTables", "Subject.csv")
S19_CODE_STATES = os.path.join(S19_RAW_DIR, "Data", "CodeStates", "CodeStates.csv")

F19_MAIN_TABLE = os.path.join(F19_RAW_DIR, "All", "Data", "MainTable.csv")
F19_SUBJECT = os.path.join(F19_RAW_DIR, "All", "Data", "LinkTables", "Subject.csv")
F19_CODE_STATES = os.path.join(F19_RAW_DIR, "All", "Data", "CodeStates", "CodeStates.csv")

# Cleaned data paths
CLEANED_ACTIVITIES_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_activities.parquet")
STUDENT_TARGETS_PATH = os.path.join(PROCESSED_DATA_DIR, "student_targets.parquet")

# Assignment stages definition (Chronological order)
ASSIGNMENT_STAGES = {
    1: [439],
    2: [439, 487],
    3: [439, 487, 492],
    4: [439, 487, 492, 494],
    5: [439, 487, 492, 494, 502],
}

# The 7 programming pattern features
FEATURE_NAMES = [
    "TimeSpent",
    "Valid",
    "CorrectSub",
    "IncorrectSub",
    "CompileError",
    "Scores",
    "EditDistance",
]

# Experiment settings
EXPERIMENT_SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
