"""
Standalone training script - runs V1 baseline and V2 improved CTGAN models
end-to-end and persists the synthetic datasets to disk.

Usage:
    python run_training.py
"""

import time
from pathlib import Path

import pandas as pd

from src.config import V1_DIR, V2_DIR
from src.preprocessing import prepare_dataset
from src.training import train_v1_baseline, train_v2_improved


def main():
    overall_start = time.time()

    print("\n" + "=" * 60)
    print("CTGAN TRAINING PIPELINE - V1 + V2")
    print("=" * 60)

    # 1. Prepare data once (reused for both models)
    print("\n[STEP 1/3] Preparing dataset...")
    df_train, df_holdout = prepare_dataset(verbose=True)

    # Persist train/holdout for downstream use
    V1_DIR.mkdir(parents=True, exist_ok=True)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(V1_DIR / "real_train.csv", index=False)
    df_holdout.to_csv(V1_DIR / "real_holdout.csv", index=False)
    print(f"\n  Saved real_train.csv and real_holdout.csv to {V1_DIR}")

    # 2. Train V1 baseline
    print("\n[STEP 2/3] Training V1 baseline...")
    model_v1, synthetic_v1, elapsed_v1 = train_v1_baseline(df_train, verbose=True)
    synthetic_v1.to_csv(V1_DIR / "synthetic.csv", index=False)
    print(f"  V1 synthetic data saved to {V1_DIR / 'synthetic.csv'}")
    print(f"  V1 training time: {elapsed_v1/60:.2f} minutes")

    # 3. Train V2 improved
    print("\n[STEP 3/3] Training V2 improved (with log-transform)...")
    model_v2, synthetic_v2, elapsed_v2 = train_v2_improved(df_train, verbose=True)
    synthetic_v2.to_csv(V2_DIR / "synthetic.csv", index=False)
    print(f"  V2 synthetic data saved to {V2_DIR / 'synthetic.csv'}")
    print(f"  V2 training time: {elapsed_v2/60:.2f} minutes")

    # Summary
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED in {overall_elapsed/60:.2f} minutes")
    print("=" * 60)
    print(f"  V1 synthetic: {V1_DIR / 'synthetic.csv'}")
    print(f"  V2 synthetic: {V2_DIR / 'synthetic.csv'}")
    print(f"  Real train:   {V1_DIR / 'real_train.csv'}")
    print(f"  Real holdout: {V1_DIR / 'real_holdout.csv'}")


if __name__ == "__main__":
    main()