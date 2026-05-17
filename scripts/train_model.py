#!/usr/bin/env python3
"""Train Face-to-BMI SVR on VGG16 fc6 embeddings."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face2bmi.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Re-extract embeddings even if cached",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel CPU workers for SVR search. Use -1 only if you want all cores.",
    )
    parser.add_argument(
        "--scoring",
        default="pearson",
        choices=["pearson", "neg_mean_absolute_error"],
        help="Model-selection metric. Pearson matches the paper's reported metric.",
    )
    args = parser.parse_args()

    meta = train_model(
        force_features=args.force_features,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        scoring=args.scoring,
    )
    print("Training complete.")
    print(f"  Best params: {meta['best_params']}")
    print(f"  Selection metric: {meta['selection_metric']}")
    print(f"  Best CV score: {meta['best_cv_score']}")
    print(f"  Train samples: {meta['train_samples']}")
    print(f"  Model saved to: {ROOT / 'models' / 'face2bmi_svr.joblib'}")


if __name__ == "__main__":
    main()
