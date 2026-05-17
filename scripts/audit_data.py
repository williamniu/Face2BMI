#!/usr/bin/env python3
"""Audit dataset and write manifests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face2bmi.data import run_audit


def main():
    train_df, test_df, summary = run_audit()
    print("Dataset audit complete.")
    print(f"  Total CSV rows: {summary['total_csv_rows']}")
    print(f"  Available images: {summary['available_images']}")
    print(f"  Missing images: {summary['missing_images']}")
    print(f"  Train (available): {summary['train_available']}")
    print(f"  Test (available): {summary['test_available']}")
    print(f"  Reports: {ROOT / 'reports'}")


if __name__ == "__main__":
    main()
