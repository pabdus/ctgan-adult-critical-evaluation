"""
Configuration module for CTGAN evaluation project.

Centralizes all constants, paths, and hyperparameters to ensure
reproducibility and easy modification of experimental settings.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
V1_DIR = RESULTS_DIR / "v1_baseline"
V2_DIR = RESULTS_DIR / "v2_improved"

# Dataset file
ADULT_CSV_PATH = RAW_DATA_DIR / "adult.csv"

# ============================================================
# REPRODUCIBILITY
# ============================================================
RANDOM_SEED = 42

# ============================================================
# DATASET CONFIGURATION
# ============================================================
TARGET_COLUMN = "income"

# Columns to exclude from modeling (sampling weight, not an attribute)
COLUMNS_TO_DROP = ["fnlwgt"]

# Missing value marker in Adult Census
MISSING_VALUE_MARKER = "?"

# Categorical columns (must be declared explicitly for CTGAN)
CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
    "income",
]

# Continuous columns (derived for clarity, all non-categorical non-dropped)
CONTINUOUS_COLUMNS = [
    "age",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

# Sample size for training (subset of full dataset for tractable training time)
TRAIN_SAMPLE_SIZE = 10_000

# ============================================================
# CTGAN HYPERPARAMETERS
# ============================================================
CTGAN_EPOCHS = 300
CTGAN_BATCH_SIZE = 500
CTGAN_USE_GPU = False

# ============================================================
# EVALUATION THRESHOLDS
# ============================================================
# Wasserstein relative thresholds (% of variable range)
WASSERSTEIN_EXCELLENT = 5.0
WASSERSTEIN_ACCEPTABLE = 15.0

# JSD thresholds
JSD_EXCELLENT = 0.05
JSD_ACCEPTABLE = 0.15

# Frobenius relative thresholds (%)
FROBENIUS_EXCELLENT = 10.0
FROBENIUS_ACCEPTABLE = 25.0

# TSTR ratio thresholds
TSTR_HIGH_UTILITY = 0.95
TSTR_MEDIUM_UTILITY = 0.85
TSTR_LOW_UTILITY = 0.70

# ============================================================
# V2 IMPROVEMENTS CONFIGURATION
# ============================================================
# Variables to apply log1p transform (zero-inflated, heavy-tailed)
LOG_TRANSFORM_COLUMNS = ["capital.gain", "capital.loss"]