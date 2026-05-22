#!/usr/bin/env python3
"""Train a small MLP regression head on cached face embeddings."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face2bmi.config import DEFAULT_BACKBONE
from face2bmi.finetune import FinetuneConfig, train_mlp_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    summary = train_mlp_head(
        FinetuneConfig(
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    )
    print("MLP head trained.")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
