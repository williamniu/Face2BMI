#!/usr/bin/env python3
"""Train Face-to-BMI: face-trained backbones + SVR/Ridge + ensemble selection."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face2bmi.config import ENSEMBLE_BACKBONES
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
        help="Parallel CPU workers for SVR/Ridge search. Use -1 only if you want all cores.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=ENSEMBLE_BACKBONES,
        choices=["vgg16_imagenet", "facenet_vggface2", "facenet_casia"],
        help="Backbones to train. Default: VGGFace2 face features + ImageNet VGG16.",
    )
    args = parser.parse_args()

    meta = train_model(
        force_features=args.force_features,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        backbones=args.backbones,
    )
    print("Training complete.")
    dep = meta["deployed"]
    print(
        f"  Deployed: ensemble of {dep['n_members']} heads from "
        f"{dep['backbones']} (test_r = {dep['test_pearson']:.4f})"
    )
    print(f"  Full ensemble test Pearson: {meta['full_ensemble_test_pearson']:.4f}")
    for entry in meta["per_backbone"]:
        print(
            f"  - {entry['backbone']:24s} "
            f"svr={entry['svr_test_pearson']:.3f}  "
            f"ridge={entry['ridge_test_pearson']:.3f}"
        )
    print(f"  Model saved to: {ROOT / 'models' / 'face2bmi_model.joblib'}")


if __name__ == "__main__":
    main()
