"""Single-image and pair inference for demo."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from face2bmi.config import INPUT_SIZE
from face2bmi.data import bmi_category
from face2bmi.features import VGGFeatureExtractor, get_preprocess
from face2bmi.train import load_trained_model


_extractor: VGGFeatureExtractor | None = None
_model = None


def _get_extractor() -> VGGFeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = VGGFeatureExtractor()
    return _extractor


def _get_model():
    global _model
    if _model is None:
        _model = load_trained_model()
    return _model


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    preprocess = get_preprocess()
    return preprocess(image.convert("RGB")).unsqueeze(0)


def extract_features_from_image(image: Image.Image) -> np.ndarray:
    tensor = image_to_tensor(image)
    extractor = _get_extractor()
    return extractor.extract_batch(tensor)


def extract_features_from_path(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return extract_features_from_image(img)


def extract_features_from_bytes(data: bytes) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB")
    return extract_features_from_image(img)


def predict_bmi_from_features(features: np.ndarray) -> float:
    model = _get_model()
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return float(model.predict(features)[0])


def predict_bmi_from_image(image: Image.Image) -> dict:
    features = extract_features_from_image(image)
    bmi = predict_bmi_from_features(features)
    return {
        "predicted_bmi": round(bmi, 2),
        "bmi_category": bmi_category(bmi),
    }


def predict_bmi_from_bytes(data: bytes) -> dict:
    img = Image.open(BytesIO(data)).convert("RGB")
    return predict_bmi_from_image(img)


def compare_pair(image_a: Image.Image, image_b: Image.Image) -> dict:
    pred_a = predict_bmi_from_image(image_a)
    pred_b = predict_bmi_from_image(image_b)
    bmi_a, bmi_b = pred_a["predicted_bmi"], pred_b["predicted_bmi"]
    if abs(bmi_a - bmi_b) < 1e-6:
        heavier = "tie"
    elif bmi_a > bmi_b:
        heavier = "A"
    else:
        heavier = "B"
    return {
        "image_a": pred_a,
        "image_b": pred_b,
        "heavier_image": heavier,
        "bmi_difference": round(abs(bmi_a - bmi_b), 2),
    }


def compare_pair_from_bytes(data_a: bytes, data_b: bytes) -> dict:
    img_a = Image.open(BytesIO(data_a)).convert("RGB")
    img_b = Image.open(BytesIO(data_b)).convert("RGB")
    return compare_pair(img_a, img_b)
