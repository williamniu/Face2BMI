"""Paper-style evaluation: Pearson r, pairwise accuracy, bias diagnostic.

Now reports per-backbone, ensemble, and the deployed model. Comparison table
versus the paper is generated as part of the report.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from face2bmi.config import MODELS_DIR, REPORTS_DIR
from face2bmi.data import bmi_category
from face2bmi.features import load_cached_embeddings
from face2bmi.train import load_trained_model


# ----------------------- core metric helpers -----------------------


def pearson_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, p = stats.pearsonr(y_true, y_pred)
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(len(y_true)),
    }


def _predict(artifact: dict, backbones_to_data: dict) -> np.ndarray:
    if artifact["type"] == "single":
        data = backbones_to_data[artifact["backbone"]]
        return artifact["estimator"].predict(data["embeddings"])
    preds = [
        m["estimator"].predict(backbones_to_data[m["backbone"]]["embeddings"])
        for m in artifact["members"]
    ]
    return np.mean(preds, axis=0)


def _regression_block(y_true, y_pred, genders) -> dict:
    overall = pearson_metrics(y_true, y_pred)
    by_gender = {}
    for g in np.unique(genders):
        mask = genders == g
        by_gender[str(g)] = pearson_metrics(y_true[mask], y_pred[mask])
    categories_true = [bmi_category(float(b)) for b in y_true]
    categories_pred = [bmi_category(float(b)) for b in y_pred]
    cat_accuracy = float(
        np.mean([t == p for t, p in zip(categories_true, categories_pred)])
    )
    return {
        "overall": overall,
        "by_gender": by_gender,
        "category_exact_match": cat_accuracy,
    }


# ----------------------- pairwise + bias (unchanged logic) -----------------------


def _gender_pair_type(g1: str, g2: str) -> str:
    g1, g2 = str(g1), str(g2)
    if g1 == g2 == "Male":
        return "male_vs_male"
    if g1 == g2 == "Female":
        return "female_vs_female"
    return "female_vs_male"


def sample_pairwise_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genders: np.ndarray,
    n_per_type: int = 300,
    n_bins: int = 15,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    indices = list(range(len(y_true)))
    by_type_bin: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            diff = abs(float(y_true[i]) - float(y_true[j]))
            if diff <= 0.5:
                continue
            ptype = _gender_pair_type(genders[i], genders[j])
            for b in range(n_bins):
                low, high = 0.5 + b, 1.5 + b
                if low < diff <= high:
                    by_type_bin[ptype][b].append((i, j))
                    break

    pairs: list[dict] = []
    target_per_bin = max(1, n_per_type // n_bins)

    for ptype in ("male_vs_male", "female_vs_female", "female_vs_male"):
        collected = 0
        for b in range(n_bins):
            pool = by_type_bin[ptype][b]
            if not pool:
                continue
            rng.shuffle(pool)
            take = min(target_per_bin, len(pool), n_per_type - collected)
            for i, j in pool[:take]:
                true_heavier = i if y_true[i] > y_true[j] else j
                pred_heavier = i if y_pred[i] > y_pred[j] else j
                pairs.append(
                    {
                        "idx_a": i,
                        "idx_b": j,
                        "pair_type": ptype,
                        "bmi_bin": b,
                        "bmi_diff": abs(float(y_true[i]) - float(y_true[j])),
                        "correct": true_heavier == pred_heavier,
                        "gender_a": str(genders[i]),
                        "gender_b": str(genders[j]),
                    }
                )
                collected += 1
                if collected >= n_per_type:
                    break
            if collected >= n_per_type:
                break
    return pairs


def _pairwise_block(y_true, y_pred, genders, n_per_type: int = 300) -> dict:
    pairs = sample_pairwise_pairs(y_true, y_pred, genders, n_per_type=n_per_type)
    if not pairs:
        return {"error": "no pairs sampled", "n_pairs": 0}

    overall_acc = float(np.mean([p["correct"] for p in pairs]))
    by_type = {}
    for ptype in ("male_vs_male", "female_vs_female", "female_vs_male"):
        subset = [p for p in pairs if p["pair_type"] == ptype]
        if subset:
            by_type[ptype] = {
                "accuracy": float(np.mean([p["correct"] for p in subset])),
                "n_pairs": len(subset),
            }
    by_bin = {}
    for b in range(15):
        subset = [p for p in pairs if p["bmi_bin"] == b]
        if subset:
            by_bin[str(b)] = {
                "accuracy": float(np.mean([p["correct"] for p in subset])),
                "n_pairs": len(subset),
                "bmi_diff_range": f"{0.5+b} < diff <= {1.5+b}",
            }
    return {
        "n_pairs": len(pairs),
        "overall_accuracy": overall_acc,
        "by_pair_type": by_type,
        "by_bmi_bin": by_bin,
    }


def _gender_bias_block(
    y_true, y_pred, genders, max_pairs: int = 2000, bmi_threshold: float = 1.0
) -> dict:
    males = [i for i, g in enumerate(genders) if g == "Male"]
    females = [i for i, g in enumerate(genders) if g == "Female"]
    rng = random.Random(42)
    rng.shuffle(males)
    rng.shuffle(females)
    pairs = []
    for mi in males:
        for fi in females:
            if abs(float(y_true[mi]) - float(y_true[fi])) >= bmi_threshold:
                continue
            pairs.append((float(y_pred[fi]) > float(y_pred[mi])))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        return {"n_pairs": 0, "note": "insufficient close male-female pairs"}
    n = len(pairs)
    n_f = sum(pairs)
    return {
        "n_pairs": n,
        "fraction_predicted_higher_female": n_f / n,
        "fraction_predicted_higher_male": 1 - n_f / n,
        "binom_p_value_two_sided": float(stats.binomtest(n_f, n, 0.5).pvalue),
        "bmi_diff_threshold": bmi_threshold,
    }


# ----------------------- top-level evaluation -----------------------


def _load_backbones_to_data(backbones: Sequence[str], split: str) -> dict:
    return {bb: load_cached_embeddings(split, backbone=bb) for bb in backbones}


def _ensemble_artifact(metadata: dict) -> dict | None:
    """Reconstruct ensemble artifact from per_backbone for evaluation,
    even if a single backbone was deployed. Used only for the comparison table."""
    return None  # not needed — we use the deployed artifact's structure directly


def run_full_evaluation(n_pairwise_per_type: int = 300) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_trained_model()
    meta_path = MODELS_DIR / "training_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if artifact["type"] == "single":
        backbones = [artifact["backbone"]]
    else:
        backbones = sorted({m["backbone"] for m in artifact["members"]})

    backbones_to_data = _load_backbones_to_data(backbones, "test")
    ref_data = next(iter(backbones_to_data.values()))
    y_true = ref_data["bmi"]
    genders = ref_data["gender"]
    names = ref_data["names"].tolist()

    y_pred = _predict(artifact, backbones_to_data)

    deployed_regression = _regression_block(y_true, y_pred, genders)
    deployed_pairwise = _pairwise_block(
        y_true, y_pred, genders, n_per_type=n_pairwise_per_type
    )
    deployed_bias = _gender_bias_block(y_true, y_pred, genders)

    # Per-backbone breakdown
    per_backbone = []
    for entry in metadata.get("per_backbone", []):
        per_backbone.append(
            {
                "backbone": entry["backbone"],
                "svr_test_pearson": entry["svr_test_pearson"],
                "ridge_test_pearson": entry["ridge_test_pearson"],
                "best_test_pearson": max(
                    entry["svr_test_pearson"], entry["ridge_test_pearson"]
                ),
            }
        )

    paper_targets = {
        "vgg_net_overall": 0.47,
        "vgg_net_male": 0.58,
        "vgg_net_female": 0.36,
        "vgg_face_overall": 0.65,
        "vgg_face_male": 0.71,
        "vgg_face_female": 0.57,
    }
    deployed_overall = deployed_regression["overall"]["pearson_r"]
    beats_paper = bool(deployed_overall > paper_targets["vgg_face_overall"])

    report = {
        "deployed": {
            "type": artifact["type"],
            "backbones": backbones,
            "regression": deployed_regression,
            "pairwise": deployed_pairwise,
            "gender_bias_close_pairs": deployed_bias,
        },
        "per_backbone": per_backbone,
        "ensemble_test_pearson": metadata.get("ensemble_test_pearson"),
        "paper_targets": paper_targets,
        "beats_paper_vgg_face_overall": beats_paper,
        "predictions": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "gender": genders.tolist(),
            "names": names,
        },
    }
    (REPORTS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2))

    summary = [
        "# Evaluation Summary",
        "",
        f"Deployed model: {artifact['type']} ({', '.join(backbones)})",
        f"Overall Pearson r: {deployed_overall:.3f}",
        (
            "Male Pearson r: "
            f"{deployed_regression['by_gender'].get('Male', {}).get('pearson_r', float('nan')):.3f}"
        ),
        (
            "Female Pearson r: "
            f"{deployed_regression['by_gender'].get('Female', {}).get('pearson_r', float('nan')):.3f}"
        ),
        f"MAE: {deployed_regression['overall']['mae']:.3f}",
        f"RMSE: {deployed_regression['overall']['rmse']:.3f}",
        f"Pairwise accuracy: {deployed_pairwise.get('overall_accuracy', 'N/A')}",
        "",
        "## Comparison vs paper",
        "",
        "| Model | Overall r | Male r | Female r |",
        "|---|---:|---:|---:|",
        f"| Paper VGG-Net | {paper_targets['vgg_net_overall']:.2f} | {paper_targets['vgg_net_male']:.2f} | {paper_targets['vgg_net_female']:.2f} |",
        f"| Paper VGG-Face | {paper_targets['vgg_face_overall']:.2f} | {paper_targets['vgg_face_male']:.2f} | {paper_targets['vgg_face_female']:.2f} |",
        (
            f"| **This run ({artifact['type']})** | "
            f"**{deployed_overall:.3f}** | "
            f"**{deployed_regression['by_gender'].get('Male', {}).get('pearson_r', float('nan')):.3f}** | "
            f"**{deployed_regression['by_gender'].get('Female', {}).get('pearson_r', float('nan')):.3f}** |"
        ),
        "",
        f"Beats paper VGG-Face overall? **{beats_paper}**",
    ]
    (REPORTS_DIR / "evaluation_summary.md").write_text("\n".join(summary))

    return report
