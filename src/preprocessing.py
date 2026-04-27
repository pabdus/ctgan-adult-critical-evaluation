"""
Preprocessing module for Adult Census Income dataset.

Handles loading, cleaning, and splitting the dataset for CTGAN training
and downstream evaluation. All functions are pure (no side effects on
global state) and take their dependencies as explicit parameters.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ADULT_CSV_PATH,
    COLUMNS_TO_DROP,
    MISSING_VALUE_MARKER,
    RANDOM_SEED,
    TARGET_COLUMN,
    TRAIN_SAMPLE_SIZE,
)


def load_adult_dataset(path=None) -> pd.DataFrame:
    """
    Load the Adult Census Income dataset from disk.

    Parameters
    ----------
    path : Path or str, optional
        Path to the CSV file. Defaults to ADULT_CSV_PATH from config.

    Returns
    -------
    pd.DataFrame
        Raw dataset with 32,561 rows and 15 columns.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the specified path.
    """
    csv_path = path or ADULT_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            f"See data/README.md for download instructions."
        )
    return pd.read_csv(csv_path)


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report columns with missing values encoded as '?'.

    Adult Census uses '?' as the missing value marker rather than NaN.
    This function detects and reports their distribution per column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset to inspect.

    Returns
    -------
    pd.DataFrame
        Summary with columns: 'column', 'missing_count', 'missing_pct'.
        Only includes columns that have at least one missing value.
    """
    rows = []
    for col in df.select_dtypes(include="object").columns:
        n_missing = (df[col] == MISSING_VALUE_MARKER).sum()
        if n_missing > 0:
            rows.append({
                "column": col,
                "missing_count": n_missing,
                "missing_pct": round(n_missing / len(df) * 100, 2),
            })
    return pd.DataFrame(rows)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Adult Census dataset.

    Performs three operations:
    1. Replace '?' with NaN and drop affected rows (~7.4% of data)
    2. Drop the 'fnlwgt' column (sampling weight, not an individual attribute)
    3. Reset index for clean downstream processing

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from load_adult_dataset().

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for sampling and modeling.
        Expected shape: ~30,162 rows × 14 columns.
    """
    df_clean = df.replace(MISSING_VALUE_MARKER, np.nan).dropna()
    df_clean = df_clean.drop(columns=COLUMNS_TO_DROP)
    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def stratified_train_holdout_split(
    df: pd.DataFrame,
    train_size: int = TRAIN_SAMPLE_SIZE,
    target_col: str = TARGET_COLUMN,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned dataset into training and holdout sets.

    The training set is used to train CTGAN. The holdout set, never seen
    by CTGAN, is reserved for unbiased evaluation via TSTR (Train on
    Synthetic, Test on Real).

    Stratification by the target column ensures both splits preserve
    the original class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from clean_dataset().
    train_size : int
        Number of rows for the training set. Defaults to 10,000.
    target_col : str
        Column name to stratify by. Defaults to 'income'.
    random_state : int
        Seed for reproducible splits.

    Returns
    -------
    df_train : pd.DataFrame
        Training set of `train_size` rows.
    df_holdout : pd.DataFrame
        Holdout set with all remaining rows.
    """
    df_train, df_holdout = train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    df_train = df_train.reset_index(drop=True)
    df_holdout = df_holdout.reset_index(drop=True)
    return df_train, df_holdout


def prepare_dataset(verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline: load, clean, and split the Adult dataset.

    Convenience function that chains the individual steps. Use this
    when you don't need intermediate access to the raw or cleaned data.

    Parameters
    ----------
    verbose : bool
        If True, print progress and shape information at each step.

    Returns
    -------
    df_train : pd.DataFrame
        Training set ready for CTGAN.
    df_holdout : pd.DataFrame
        Holdout set reserved for TSTR evaluation.
    """
    if verbose:
        print("Loading Adult Census dataset...")
    df_raw = load_adult_dataset()

    if verbose:
        print(f"  Raw shape: {df_raw.shape}")
        print("\nCleaning dataset...")
    df_clean = clean_dataset(df_raw)

    if verbose:
        rows_dropped = df_raw.shape[0] - df_clean.shape[0]
        pct_dropped = rows_dropped / df_raw.shape[0] * 100
        print(f"  Cleaned shape: {df_clean.shape}")
        print(f"  Rows dropped: {rows_dropped} ({pct_dropped:.2f}%)")
        print("\nSplitting train/holdout...")

    df_train, df_holdout = stratified_train_holdout_split(df_clean)

    if verbose:
        print(f"  Train: {df_train.shape}")
        print(f"  Holdout: {df_holdout.shape}")
        train_dist = df_train[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict()
        holdout_dist = df_holdout[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict()
        print(f"  Train target distribution:   {train_dist}")
        print(f"  Holdout target distribution: {holdout_dist}")

    return df_train, df_holdout