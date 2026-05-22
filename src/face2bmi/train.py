"""Train regression heads on cached embeddings.

For each backbone we fit BOTH an SVR head and a Ridge head (small grids).
The deployed model is an unweighted average of all heads belonging to the
configured `DEPLOY_BACKBONES` — this consistently beats any single head and
is robust to the per-backbone "best head" choice flipping with random seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from face2bmi.config import (
    BACKBONES,
    DEPLOY_BACKBONES,
    ENSEMBLE_BACKBONES,
    MODELS_DIR,
)
from face2bmi.data import run_audit
from face2bmi.features import cache_split_embeddings, load_cached_embeddings


def build_svr_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])


def build_ridge_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])


def pearson_corr_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(stats.pearsonr(y_true, y_pred)[0])


PEARSON_SCORER = make_scorer(pearson_corr_score)


def _search_svr(X: np.ndarray, y: np.ndarray, cv: int, n_jobs: int) -> GridSearchCV:
    grid = {
        "svr__C": [10.0, 30.0, 100.0],
        "svr__epsilon": [0.01, 0.1],
        "svr__gamma": ["scale", 1e-5, 1e-4],
    }
    search = GridSearchCV(
        build_svr_pipeline(), grid, cv=cv, scoring=PEARSON_SCORER, n_jobs=n_jobs
    )
    return search.fit(X, y)


def _search_ridge(X: np.ndarray, y: np.ndarray, cv: int, n_jobs: int) -> GridSearchCV:
    grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
    search = GridSearchCV(
        build_ridge_pipeline(), grid, cv=cv, scoring=PEARSON_SCORER, n_jobs=n_jobs
    )
    return search.fit(X, y)


def _train_one_backbone(
    backbone: str,
    train_df,
    test_df,
    force_features: bool,
    cv: int,
    n_jobs: int,
) -> dict:
    cache_split_embeddings(train_df, "train", backbone=backbone, force=force_features)
    cache_split_embeddings(test_df, "test", backbone=backbone, force=force_features)

    train_data = load_cached_embeddings("train", backbone=backbone)
    X_train, y_train = train_data["embeddings"], train_data["bmi"]

    test_data = load_cached_embeddings("test", backbone=backbone)
    X_test, y_test = test_data["embeddings"], test_data["bmi"]

    svr = _search_svr(X_train, y_train, cv, n_jobs)
    ridge = _search_ridge(X_train, y_train, cv, n_jobs)

    svr_test = pearson_corr_score(y_test, svr.predict(X_test))
    ridge_test = pearson_corr_score(y_test, ridge.predict(X_test))

    return {
        "backbone": backbone,
        "svr_estimator": svr.best_estimator_,
        "ridge_estimator": ridge.best_estimator_,
        "svr_best_params": svr.best_params_,
        "ridge_best_params": ridge.best_params_,
        "svr_cv_score": float(svr.best_score_),
        "ridge_cv_score": float(ridge.best_score_),
        "svr_test_pearson": float(svr_test),
        "ridge_test_pearson": float(ridge_test),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
    }


def _ensemble_predict(
    estimators_with_backbones: Sequence[tuple[str, object]],
    backbones_to_data: dict,
) -> np.ndarray:
    preds = [
        est.predict(backbones_to_data[bb]["embeddings"])
        for bb, est in estimators_with_backbones
    ]
    return np.mean(preds, axis=0)


def train_model(
    force_features: bool = False,
    cv_folds: int = 3,
    n_jobs: int = 1,
    backbones: Sequence[str] | None = None,
    deploy_backbones: Sequence[str] | None = None,
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df, audit = run_audit()
    backbones = list(backbones or ENSEMBLE_BACKBONES)
    deploy_backbones = list(deploy_backbones or DEPLOY_BACKBONES)

    per_backbone: list[dict] = []
    for name in backbones:
        per_backbone.append(
            _train_one_backbone(
                name, train_df, test_df, force_features, cv_folds, n_jobs
            )
        )

    backbones_to_data = {
        bb: load_cached_embeddings("test", backbone=bb) for bb in backbones
    }
    y_test = backbones_to_data[backbones[0]]["bmi"]

    # All-backbone × both-heads ensemble (informational).
    all_pairs = []
    for r in per_backbone:
        all_pairs.append((r["backbone"], r["svr_estimator"]))
        all_pairs.append((r["backbone"], r["ridge_estimator"]))
    full_ensemble_pred = _ensemble_predict(all_pairs, backbones_to_data)
    full_ensemble_pearson = pearson_corr_score(y_test, full_ensemble_pred)

    # Deployed: only the backbones we want to ship + both heads.
    deploy_pairs = []
    for r in per_backbone:
        if r["backbone"] in deploy_backbones:
            deploy_pairs.append((r["backbone"], r["svr_estimator"]))
            deploy_pairs.append((r["backbone"], r["ridge_estimator"]))
    deploy_pred = _ensemble_predict(deploy_pairs, backbones_to_data)
    deploy_pearson = pearson_corr_score(y_test, deploy_pred)

    # Save the deployed ensemble.
    deploy_payload = {
        "type": "ensemble",
        "backbones": deploy_backbones,
        "members": [
            {"backbone": bb, "estimator": est} for bb, est in deploy_pairs
        ],
    }
    joblib.dump(deploy_payload, MODELS_DIR / "face2bmi_model.joblib")

    metadata = {
        "deployed": {
            "type": "ensemble",
            "backbones": deploy_backbones,
            "heads_per_backbone": ["svr", "ridge"],
            "n_members": len(deploy_pairs),
            "test_pearson": float(deploy_pearson),
        },
        "full_ensemble_test_pearson": float(full_ensemble_pearson),
        "deployed_test_pearson": float(deploy_pearson),
        # Keep the legacy field name so figures/tests that read it keep working.
        "ensemble_test_pearson": float(deploy_pearson),
        "per_backbone": [
            {k: v for k, v in r.items() if not k.endswith("_estimator")}
            for r in per_backbone
        ],
        "audit": {
            "total_csv_rows": audit["total_csv_rows"],
            "available_images": audit["available_images"],
            "missing_images": audit["missing_images"],
            "train_available": audit["train_available"],
            "test_available": audit["test_available"],
        },
    }
    (MODELS_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load_trained_model():
    path = MODELS_DIR / "face2bmi_model.joblib"
    if not path.exists():
        legacy = MODELS_DIR / "face2bmi_svr.joblib"
        if legacy.exists():
            return {"type": "single", "backbone": "vgg16_imagenet", "estimator": joblib.load(legacy)}
        raise FileNotFoundError(f"Model not found at {path}. Run train_model first.")
    return joblib.load(path)
