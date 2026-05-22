#!/usr/bin/env python3
"""Run full evaluation and write reports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face2bmi.evaluate import run_full_evaluation


def main():
    report = run_full_evaluation()
    reg = report["deployed"]["regression"]
    print("Evaluation complete.")
    print(f"  Deployed: {report['deployed']['type']} ({report['deployed']['backbones']})")
    print(f"  Overall Pearson r: {reg['overall']['pearson_r']:.3f}")
    for g, m in reg["by_gender"].items():
        print(f"  {g} Pearson r: {m['pearson_r']:.3f}")
    print(
        f"  Pairwise accuracy: {report['deployed']['pairwise'].get('overall_accuracy', 'N/A')}"
    )
    print(f"  Beats paper VGG-Face overall? {report['beats_paper_vgg_face_overall']}")
    print(f"  Report: {ROOT / 'reports' / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
