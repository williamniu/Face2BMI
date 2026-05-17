"""Train SVR on cached VGG16 fc6 embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from face2bmi.config import MODELS_DIR
from face2bmi.data import run_audit
from face2bmi.features import cache_split_embeddings, load_cached_embeddings


def build_svr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf")),
        ]
    )


def pearson_corr_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score models with the same correlation metric reported in the paper."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(stats.pearsonr(y_true, y_pred)[0])


def train_model(
    force_features: bool = False,
    cv_folds: int = 3,
    n_jobs: int = 1,
    scoring: str = "pearson",
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df, audit = run_audit()

    cache_split_embeddings(train_df, "train", force=force_features)
    cache_split_embeddings(test_df, "test", force=force_features)

    train_data = load_cached_embeddings("train")
    X_train = train_data["embeddings"]
    y_train = train_data["bmi"]

    pipeline = build_svr_pipeline()
    param_grid = {
        "svr__C": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        "svr__epsilon": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
        "svr__gamma": ["scale", "auto", 1e-5, 3e-5, 1e-4, 3e-4],
    }
    scorer = make_scorer(pearson_corr_score) if scoring == "pearson" else scoring

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring=scorer,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    model_path = MODELS_DIR / "face2bmi_svr.joblib"
    joblib.dump(best, model_path)

    metadata = {
        "feature_extractor": "torchvision.vgg16_imagenet_fc6",
        "feature_dim": int(X_train.shape[1]),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(test_df)),
        "selection_metric": scoring,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "audit": {
            "total_csv_rows": audit["total_csv_rows"],
            "available_images": audit["available_images"],
            "missing_images": audit["missing_images"],
        },
    }
    meta_path = MODELS_DIR / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_trained_model():
    path = MODELS_DIR / "face2bmi_svr.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train_model first.")
    return joblib.load(path)
