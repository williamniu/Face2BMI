"""Single-image and pair inference for the demo.

Loads the deployed model artifact (single backbone or ensemble), extracts
embeddings on demand, and applies optional MTCNN face alignment for uploads.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from face2bmi.align import align_face
from face2bmi.config import BACKBONES, DEFAULT_BACKBONE
from face2bmi.data import bmi_category
from face2bmi.features import build_backbone, get_preprocess
from face2bmi.train import load_trained_model


@lru_cache(maxsize=4)
def _extractor(backbone: str):
    return build_backbone(backbone)


@lru_cache(maxsize=1)
def _model():
    return load_trained_model()


@lru_cache(maxsize=4)
def _preprocess(backbone: str):
    return get_preprocess(backbone)


def _backbone_input_size(backbone: str) -> int:
    return BACKBONES[backbone]["input_size"]


def _featurize(image: Image.Image, backbone: str, align: bool) -> np.ndarray:
    if align:
        image = align_face(image, image_size=_backbone_input_size(backbone))
    tensor = _preprocess(backbone)(image.convert("RGB")).unsqueeze(0)
    device = next(_extractor(backbone).backbone.parameters()).device
    with torch.inference_mode():
        out = _extractor(backbone)(tensor.to(device))
    return out.detach().cpu().numpy()


def _predict_bmi(image: Image.Image, align: bool = True) -> float:
    artifact = _model()
    if artifact["type"] == "single":
        feats = _featurize(image, artifact["backbone"], align=align)
        return float(artifact["estimator"].predict(feats)[0])
    # ensemble: featurize each unique backbone once, then average all member heads
    backbones = sorted({m["backbone"] for m in artifact["members"]})
    feats_by_bb = {bb: _featurize(image, bb, align=align) for bb in backbones}
    preds = [
        m["estimator"].predict(feats_by_bb[m["backbone"]])[0]
        for m in artifact["members"]
    ]
    return float(np.mean(preds))


def predict_bmi_from_image(image: Image.Image, align: bool = True) -> dict:
    bmi = _predict_bmi(image, align=align)
    return {
        "predicted_bmi": round(bmi, 2),
        "bmi_category": bmi_category(bmi),
        "model_type": _model()["type"],
        "aligned": align,
    }


def predict_bmi_from_bytes(data: bytes, align: bool = True) -> dict:
    img = Image.open(BytesIO(data)).convert("RGB")
    return predict_bmi_from_image(img, align=align)


def predict_bmi_from_path(path: str | Path, align: bool = True) -> dict:
    img = Image.open(path).convert("RGB")
    return predict_bmi_from_image(img, align=align)


def compare_pair(
    image_a: Image.Image, image_b: Image.Image, align: bool = True
) -> dict:
    pred_a = predict_bmi_from_image(image_a, align=align)
    pred_b = predict_bmi_from_image(image_b, align=align)
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


def compare_pair_from_bytes(
    data_a: bytes, data_b: bytes, align: bool = True
) -> dict:
    img_a = Image.open(BytesIO(data_a)).convert("RGB")
    img_b = Image.open(BytesIO(data_b)).convert("RGB")
    return compare_pair(img_a, img_b, align=align)


# Legacy helpers kept for any external callers.
def extract_features_from_image(image: Image.Image) -> np.ndarray:
    return _featurize(image, DEFAULT_BACKBONE, align=False)


def extract_features_from_bytes(data: bytes) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB")
    return _featurize(img, DEFAULT_BACKBONE, align=False)
