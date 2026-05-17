"""Paper-style evaluation: Pearson r, pairwise accuracy, bias diagnostic."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from face2bmi.config import MODELS_DIR, REPORTS_DIR
from face2bmi.data import bmi_category
from face2bmi.features import load_cached_embeddings
from face2bmi.train import load_trained_model


def pearson_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, p = stats.pearsonr(y_true, y_pred)
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(len(y_true)),
    }


def evaluate_regression(model, split: str = "test") -> dict:
    data = load_cached_embeddings(split)
    X, y_true = data["embeddings"], data["bmi"]
    genders = data["gender"]
    y_pred = model.predict(X)

    overall = pearson_metrics(y_true, y_pred)
    by_gender = {}
    for g in np.unique(genders):
        mask = genders == g
        by_gender[str(g)] = pearson_metrics(y_true[mask], y_pred[mask])

  # category breakdown on predictions
    categories_true = [bmi_category(float(b)) for b in y_true]
    categories_pred = [bmi_category(float(b)) for b in y_pred]
    cat_accuracy = float(
        np.mean([t == p for t, p in zip(categories_true, categories_pred)])
    )

    return {
        "split": split,
        "overall": overall,
        "by_gender": by_gender,
        "category_exact_match": cat_accuracy,
        "predictions": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "gender": genders.tolist(),
            "names": data["names"].tolist(),
        },
    }


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
    """Sample pairs stratified by gender type and BMI difference bins (paper Sec 7.2)."""
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


def evaluate_pairwise(model, split: str = "test", n_per_type: int = 300) -> dict:
    data = load_cached_embeddings(split)
    y_pred = model.predict(data["embeddings"])
    y_true = data["bmi"]
    genders = data["gender"]

    pairs = sample_pairwise_pairs(y_true, y_pred, genders, n_per_type=n_per_type)
    if not pairs:
        return {"error": "no pairs sampled", "n_pairs": 0}

    overall_acc = float(np.mean([p["correct"] for p in pairs]))
    by_type: dict[str, dict] = {}
    for ptype in ("male_vs_male", "female_vs_female", "female_vs_male"):
        subset = [p for p in pairs if p["pair_type"] == ptype]
        if subset:
            by_type[ptype] = {
                "accuracy": float(np.mean([p["correct"] for p in subset])),
                "n_pairs": len(subset),
            }

    by_bin: dict[str, dict] = {}
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


def evaluate_gender_bias(
    model,
    split: str = "test",
    max_pairs: int = 2000,
    bmi_threshold: float = 1.0,
    seed: int = 42,
) -> dict:
    """Close-BMI male-female pairs: fraction predicted higher for females (paper Sec 8.3)."""
    data = load_cached_embeddings(split)
    y_pred = model.predict(data["embeddings"])
    y_true = data["bmi"]
    genders = data["gender"]

    males = [i for i, g in enumerate(genders) if g == "Male"]
    females = [i for i, g in enumerate(genders) if g == "Female"]

    rng = random.Random(seed)
    pairs = []
    rng.shuffle(males)
    rng.shuffle(females)

    for mi in males:
        for fi in females:
            if abs(float(y_true[mi]) - float(y_true[fi])) >= bmi_threshold:
                continue
            pred_m = float(y_pred[mi])
            pred_f = float(y_pred[fi])
            higher_female = pred_f > pred_m
            pairs.append(
                {
                    "male_idx": mi,
                    "female_idx": fi,
                    "true_bmi_m": float(y_true[mi]),
                    "true_bmi_f": float(y_true[fi]),
                    "pred_higher_female": higher_female,
                }
            )
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        return {"n_pairs": 0, "note": "insufficient close male-female pairs"}

    n_female_higher = sum(1 for p in pairs if p["pred_higher_female"])
    n = len(pairs)
    # Binomial test vs 50-50
    p_value = float(stats.binomtest(n_female_higher, n, 0.5).pvalue)

    return {
        "n_pairs": n,
        "fraction_predicted_higher_female": n_female_higher / n,
        "fraction_predicted_higher_male": 1 - n_female_higher / n,
        "binom_p_value_two_sided": p_value,
        "bmi_diff_threshold": bmi_threshold,
        "interpretation": (
            "Values near 0.5 suggest no strong gender bias in close-BMI pairs."
        ),
    }


def run_full_evaluation(n_pairwise_per_type: int = 300) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_trained_model()
    regression = evaluate_regression(model, "test")
    pairwise = evaluate_pairwise(model, "test", n_per_type=n_pairwise_per_type)
    bias = evaluate_gender_bias(model, "test")

    paper_targets = {
        "vgg_face_overall_pearson": 0.65,
        "vgg_face_male_pearson": 0.71,
        "vgg_face_female_pearson": 0.57,
        "note": "Paper used VGG-Face fc6; this reproduction uses VGG16 ImageNet fc6.",
    }

    report = {
        "regression": regression,
        "pairwise": pairwise,
        "gender_bias_close_pairs": bias,
        "paper_targets": paper_targets,
    }

    out_path = REPORTS_DIR / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    summary_lines = [
        "# Evaluation Summary",
        "",
        f"Overall Pearson r: {regression['overall']['pearson_r']:.3f}",
        f"Male Pearson r: {regression['by_gender'].get('Male', {}).get('pearson_r', 'N/A')}",
        f"Female Pearson r: {regression['by_gender'].get('Female', {}).get('pearson_r', 'N/A')}",
        f"MAE: {regression['overall']['mae']:.3f}",
        f"RMSE: {regression['overall']['rmse']:.3f}",
        "",
        f"Pairwise accuracy: {pairwise.get('overall_accuracy', 'N/A')}",
        f"Gender bias (close pairs): {bias}",
    ]
    (REPORTS_DIR / "evaluation_summary.md").write_text("\n".join(summary_lines))

    return report
