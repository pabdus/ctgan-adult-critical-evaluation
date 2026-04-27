"""
Training module for CTGAN models.

Encapsulates CTGAN initialization, training, and synthetic sample
generation. Supports two variants:
- v1_baseline: standard CTGAN on raw cleaned data
- v2_improved: CTGAN with log1p transformation on zero-inflated variables

Both variants share the same training infrastructure but differ in
preprocessing applied before model.fit().
"""

import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN

from src.config import (
    CATEGORICAL_COLUMNS,
    CTGAN_BATCH_SIZE,
    CTGAN_EPOCHS,
    CTGAN_USE_GPU,
    LOG_TRANSFORM_COLUMNS,
    RANDOM_SEED,
)


# ============================================================
# CORE TRAINING
# ============================================================

def set_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Fix random seeds for reproducibility across NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value. Defaults to RANDOM_SEED from config.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_ctgan(
    df: pd.DataFrame,
    discrete_columns: list = None,
    epochs: int = CTGAN_EPOCHS,
    batch_size: int = CTGAN_BATCH_SIZE,
    use_gpu: bool = CTGAN_USE_GPU,
    verbose: bool = True,
) -> Tuple[CTGAN, float]:
    """
    Train a CTGAN model on tabular data.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (real samples).
    discrete_columns : list, optional
        Names of categorical columns. Defaults to CATEGORICAL_COLUMNS.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for adversarial training.
    use_gpu : bool
        Whether to enable GPU acceleration.
    verbose : bool
        If True, print training progress.

    Returns
    -------
    model : CTGAN
        Fitted CTGAN model.
    elapsed_seconds : float
        Total training time in seconds.
    """
    if discrete_columns is None:
        discrete_columns = CATEGORICAL_COLUMNS

    set_seeds()

    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        cuda=use_gpu,
    )

    if verbose:
        print(f"Training CTGAN on {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Discrete columns: {len(discrete_columns)} | Continuous: {df.shape[1] - len(discrete_columns)}")
        print(f"Epochs: {epochs} | Batch size: {batch_size}")
        print("-" * 60)

    start = time.time()
    model.fit(df, discrete_columns=discrete_columns)
    elapsed = time.time() - start

    if verbose:
        print("-" * 60)
        print(f"Training completed in {elapsed/60:.2f} minutes")

    return model, elapsed


def generate_synthetic(
    model: CTGAN,
    n_samples: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic samples from a trained CTGAN model.

    Following standard practice in the literature, n_samples should
    typically equal the size of the real training set for balanced
    metric comparison.

    Parameters
    ----------
    model : CTGAN
        Trained CTGAN model.
    n_samples : int
        Number of synthetic rows to generate.
    verbose : bool
        If True, print generation progress.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with the same schema as the training data.
    """
    set_seeds()

    if verbose:
        print(f"Generating {n_samples} synthetic samples...")

    start = time.time()
    synthetic = model.sample(n_samples)
    elapsed = time.time() - start

    if verbose:
        print(f"Generation completed in {elapsed:.1f} seconds")

    return synthetic


# ============================================================
# V2 IMPROVEMENTS: LOG TRANSFORMATION
# ============================================================

def apply_log_transform(
    df: pd.DataFrame,
    columns: list = None,
) -> pd.DataFrame:
    """
    Apply log1p transformation to specified columns.

    Used in v2 to compress the heavy tails of zero-inflated variables
    (capital.gain, capital.loss) before CTGAN training. log1p = log(1+x)
    handles zeros gracefully without producing -inf.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list, optional
        Columns to transform. Defaults to LOG_TRANSFORM_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Copy of df with specified columns log-transformed.
    """
    if columns is None:
        columns = LOG_TRANSFORM_COLUMNS

    df_transformed = df.copy()
    for col in columns:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed


def invert_log_transform(
    df: pd.DataFrame,
    columns: list = None,
) -> pd.DataFrame:
    """
    Reverse the log1p transformation.

    Applied to synthetic samples generated from a model trained on
    log-transformed data, returning them to the original scale.

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic dataset on the log-transformed scale.
    columns : list, optional
        Columns to invert. Defaults to LOG_TRANSFORM_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Copy of df with specified columns back on the original scale.
        Values are clipped at 0 to handle minor negative outputs from CTGAN.
    """
    if columns is None:
        columns = LOG_TRANSFORM_COLUMNS

    df_inverted = df.copy()
    for col in columns:
        if col in df_inverted.columns:
            # expm1 = exp(x) - 1, the exact inverse of log1p
            # Clip negative values that may emerge from CTGAN's continuous output
            df_inverted[col] = np.expm1(df_inverted[col]).clip(lower=0)
            # Round to integer (capital.gain/loss are originally integers)
            df_inverted[col] = df_inverted[col].round().astype(int)
    return df_inverted


# ============================================================
# HIGH-LEVEL PIPELINES
# ============================================================

def train_v1_baseline(
    df_train: pd.DataFrame,
    epochs: int = CTGAN_EPOCHS,
    verbose: bool = True,
) -> Tuple[CTGAN, pd.DataFrame, float]:
    """
    V1 pipeline: standard CTGAN on the cleaned dataset.

    Parameters
    ----------
    df_train : pd.DataFrame
        Cleaned training data.
    epochs : int
        Training epochs.
    verbose : bool
        Print progress.

    Returns
    -------
    model : CTGAN
        Trained model.
    synthetic : pd.DataFrame
        Synthetic samples (same size as df_train).
    elapsed : float
        Training time in seconds.
    """
    if verbose:
        print("=" * 60)
        print("V1 BASELINE: CTGAN on raw cleaned data")
        print("=" * 60)

    model, elapsed = train_ctgan(df_train, epochs=epochs, verbose=verbose)
    synthetic = generate_synthetic(model, n_samples=len(df_train), verbose=verbose)
    return model, synthetic, elapsed


def train_v2_improved(
    df_train: pd.DataFrame,
    epochs: int = CTGAN_EPOCHS,
    verbose: bool = True,
) -> Tuple[CTGAN, pd.DataFrame, float]:
    """
    V2 pipeline: CTGAN with log1p transformation on zero-inflated columns.

    Pre-applies log1p to capital.gain and capital.loss to compress
    heavy tails. After generation, inverts the transformation to return
    synthetic samples to the original scale.

    Parameters
    ----------
    df_train : pd.DataFrame
        Cleaned training data.
    epochs : int
        Training epochs.
    verbose : bool
        Print progress.

    Returns
    -------
    model : CTGAN
        Trained model (works on log-transformed scale internally).
    synthetic : pd.DataFrame
        Synthetic samples on the original scale (after inverse transform).
    elapsed : float
        Training time in seconds.
    """
    if verbose:
        print("=" * 60)
        print("V2 IMPROVED: CTGAN with log1p on zero-inflated variables")
        print("=" * 60)
        print(f"Applying log1p to: {LOG_TRANSFORM_COLUMNS}")

    df_train_log = apply_log_transform(df_train)
    model, elapsed = train_ctgan(df_train_log, epochs=epochs, verbose=verbose)
    synthetic_log = generate_synthetic(model, n_samples=len(df_train), verbose=verbose)
    synthetic = invert_log_transform(synthetic_log)

    if verbose:
        print(f"Inverse transform applied to: {LOG_TRANSFORM_COLUMNS}")

    return model, synthetic, elapsed