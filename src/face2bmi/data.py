"""Dataset loading, auditing, and manifest generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from face2bmi.config import (
    BMI_CATEGORIES,
    DATA_CSV,
    IMAGES_DIR,
    MANIFESTS_DIR,
    REPORTS_DIR,
)


REQUIRED_COLUMNS = {"bmi", "gender", "is_training", "name"}


def bmi_category(bmi: float) -> str:
    for label, low, high in BMI_CATEGORIES:
        if low < bmi <= high:
            return label
    return "Other"


def load_raw_csv(csv_path: Path | None = None) -> pd.DataFrame:
    csv_path = csv_path or DATA_CSV
    df = pd.read_csv(csv_path)
    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "id"})
    elif "id" not in df.columns and df.columns[0] not in REQUIRED_COLUMNS:
        df = df.rename(columns={df.columns[0]: "id"})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["bmi"] = df["bmi"].astype(float)
    df["is_training"] = df["is_training"].astype(int)
    df["gender"] = df["gender"].astype(str)
    df["image_path"] = df["name"].apply(lambda n: str(IMAGES_DIR / n))
    df["image_exists"] = df["name"].apply(lambda n: (IMAGES_DIR / n).is_file())
    df["bmi_category"] = df["bmi"].apply(bmi_category)
    return df


def filter_available(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["image_exists"]].copy().reset_index(drop=True)


def summarize_df(df: pd.DataFrame, label: str) -> dict:
    return {
        "label": label,
        "n_rows": len(df),
        "is_training": dict(Counter(df["is_training"].astype(str))),
        "gender": dict(Counter(df["gender"])),
        "bmi_category": dict(Counter(df["bmi_category"])),
        "bmi_mean": float(df["bmi"].mean()),
        "bmi_std": float(df["bmi"].std()),
        "bmi_min": float(df["bmi"].min()),
        "bmi_max": float(df["bmi"].max()),
    }


def run_audit(
    csv_path: Path | None = None,
    images_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Audit dataset and write manifests + summary JSON."""
    csv_path = csv_path or DATA_CSV
    images_dir = images_dir or IMAGES_DIR
    reports_dir = reports_dir or REPORTS_DIR
    manifests_dir = reports_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_csv(csv_path)
    available = filter_available(df)
    missing = df[~df["image_exists"]]

    train_manifest = available[available["is_training"] == 1].copy()
    test_manifest = available[available["is_training"] == 0].copy()

    train_manifest.to_csv(manifests_dir / "train_manifest.csv", index=False)
    test_manifest.to_csv(manifests_dir / "test_manifest.csv", index=False)
    available.to_csv(manifests_dir / "full_manifest.csv", index=False)
    missing[["name", "bmi", "gender", "is_training"]].to_csv(
        manifests_dir / "missing_images.csv", index=False
    )

    summary = {
        "csv_path": str(csv_path),
        "images_dir": str(images_dir),
        "total_csv_rows": len(df),
        "available_images": len(available),
        "missing_images": len(missing),
        "train_available": len(train_manifest),
        "test_available": len(test_manifest),
        "raw": summarize_df(df, "all_csv_rows"),
        "available": summarize_df(available, "available_only"),
        "train": summarize_df(train_manifest, "train_available"),
        "test": summarize_df(test_manifest, "test_available"),
    }

    with open(reports_dir / "dataset_audit.json", "w") as f:
        json.dump(summary, f, indent=2)

    return train_manifest, test_manifest, summary


def load_manifest(split: str) -> pd.DataFrame:
    """Load train or test manifest (runs audit if missing)."""
    path = MANIFESTS_DIR / f"{split}_manifest.csv"
    if not path.exists():
        run_audit()
    return pd.read_csv(path)
