#!/usr/bin/env python3
"""Regenerate report figures from the latest evaluation report."""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

from face2bmi.data import load_manifest  # noqa: E402


def _load_report() -> dict:
    return json.loads((REPORTS / "evaluation_report.json").read_text())


def _load_metadata() -> dict:
    return json.loads((ROOT / "models" / "training_metadata.json").read_text())


def paper_metric_comparison(report: dict):
    paper = report["paper_targets"]
    deployed = report["deployed"]["regression"]
    overall = deployed["overall"]["pearson_r"]
    male = deployed["by_gender"].get("Male", {}).get("pearson_r", float("nan"))
    female = deployed["by_gender"].get("Female", {}).get("pearson_r", float("nan"))

    labels = ["Overall", "Male", "Female"]
    paper_net = [paper["vgg_net_overall"], paper["vgg_net_male"], paper["vgg_net_female"]]
    paper_face = [paper["vgg_face_overall"], paper["vgg_face_male"], paper["vgg_face_female"]]
    ours = [overall, male, female]

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, paper_net, width, label="Paper VGG-Net (0.47/0.58/0.36)")
    ax.bar(x, paper_face, width, label="Paper VGG-Face (0.65/0.71/0.57)")
    ax.bar(
        x + width,
        ours,
        width,
        label=f"Ours ({report['deployed']['type']}, "
        f"{', '.join(report['deployed']['backbones'])})",
        color="#2a9d8f",
    )
    for xi, v in zip(x + width, ours):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, max(0.9, max(ours) + 0.1))
    ax.set_title("Face-to-BMI: Pearson r vs paper")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper_metric_comparison.png", dpi=160)
    plt.close(fig)


def predicted_vs_actual(report: dict):
    preds = report["predictions"]
    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])
    genders = np.array(preds["gender"])
    fig, ax = plt.subplots(figsize=(6, 6))
    for g, color in [("Male", "#1d4ed8"), ("Female", "#be185d")]:
        mask = genders == g
        ax.scatter(y_true[mask], y_pred[mask], s=10, alpha=0.5, label=g, color=color)
    lim = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lim, lim, "k--", linewidth=1, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True BMI")
    ax.set_ylabel("Predicted BMI")
    ax.set_title("Predicted vs actual BMI (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=160)
    plt.close(fig)


def residuals_by_bmi(report: dict):
    preds = report["predictions"]
    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_true, residuals, s=10, alpha=0.5, color="#475569")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("True BMI")
    ax.set_ylabel("Predicted - True")
    ax.set_title("Residuals by true BMI")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals_by_bmi.png", dpi=160)
    plt.close(fig)


def pairwise_accuracy_by_bin(report: dict):
    pw = report["deployed"]["pairwise"]
    by_bin = pw.get("by_bmi_bin", {})
    if not by_bin:
        return
    bins = sorted(int(k) for k in by_bin.keys())
    accs = [by_bin[str(b)]["accuracy"] for b in bins]
    labels = [by_bin[str(b)]["bmi_diff_range"].replace(" < diff <= ", "–") for b in bins]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(bins)), accs, color="#2a9d8f")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pairwise accuracy")
    ax.set_title("Pairwise accuracy by absolute BMI difference")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pairwise_accuracy_by_bin.png", dpi=160)
    plt.close(fig)


def backbone_comparison(metadata: dict):
    entries = metadata.get("per_backbone", [])
    if not entries:
        return
    names = [e["backbone"] for e in entries]
    svr = [e["svr_test_pearson"] for e in entries]
    ridge = [e["ridge_test_pearson"] for e in entries]
    ensemble = metadata.get("ensemble_test_pearson")

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, svr, width, label="SVR head")
    ax.bar(x + width / 2, ridge, width, label="Ridge head")
    if ensemble is not None:
        ax.axhline(
            ensemble,
            color="#2a9d8f",
            linestyle="--",
            label=f"Ensemble = {ensemble:.3f}",
        )
    ax.axhline(0.65, color="#94a3b8", linestyle=":", label="Paper VGG-Face = 0.65")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Test Pearson r")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-backbone regression heads + ensemble")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "backbone_comparison.png", dpi=160)
    plt.close(fig)


def bmi_category_distribution():
    try:
        train = load_manifest("train")
        test = load_manifest("test")
    except Exception:
        return
    cats = [c[0] for c in __import__("face2bmi.config", fromlist=["BMI_CATEGORIES"]).BMI_CATEGORIES]
    train_counts = Counter(train["bmi_category"])
    test_counts = Counter(test["bmi_category"])
    train_vals = [train_counts.get(c, 0) for c in cats]
    test_vals = [test_counts.get(c, 0) for c in cats]

    x = np.arange(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, train_vals, width, label="Train")
    ax.bar(x + width / 2, test_vals, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("BMI category distribution (available images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bmi_category_distribution.png", dpi=160)
    plt.close(fig)


def main():
    report = _load_report()
    meta = _load_metadata()
    paper_metric_comparison(report)
    predicted_vs_actual(report)
    residuals_by_bmi(report)
    pairwise_accuracy_by_bin(report)
    backbone_comparison(meta)
    bmi_category_distribution()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
