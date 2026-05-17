"""FastAPI demo server for Face-to-BMI."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face2bmi.config import IMAGES_DIR, MODELS_DIR
from face2bmi.data import load_manifest
from face2bmi.inference import (
    compare_pair_from_bytes,
    predict_bmi_from_bytes,
)

STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(
    title="Face-to-BMI Demo",
    description="Educational demo — predictions are noisy at individual level.",
    version="0.1.0",
)


class SampleInfo(BaseModel):
    name: str
    bmi: float
    gender: str
    url: str


@app.on_event("startup")
def check_model():
    model_path = MODELS_DIR / "face2bmi_svr.joblib"
    if not model_path.exists():
        print(
            "WARNING: Trained model not found. Run: python scripts/train_model.py",
            file=sys.stderr,
        )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": (MODELS_DIR / "face2bmi_svr.joblib").exists(),
    }


@app.get("/api/samples", response_model=list[SampleInfo])
def list_samples(n: int = 12):
    try:
        test_df = load_manifest("test")
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    if len(test_df) == 0:
        return []
    n = min(n, len(test_df))
    rows = test_df.sample(n=n, random_state=42)
    return [
        SampleInfo(
            name=row["name"],
            bmi=float(row["bmi"]),
            gender=row["gender"],
            url=f"/api/sample-image/{row['name']}",
        )
        for _, row in rows.iterrows()
    ]


@app.get("/api/sample-image/{filename}")
def sample_image(filename: str):
    path = IMAGES_DIR / filename
    if not path.is_file() or ".." in filename:
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/bmp")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        return predict_bmi_from_bytes(data)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}") from e


@app.post("/api/compare")
async def compare(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    data_a = await file_a.read()
    data_b = await file_b.read()
    if not data_a or not data_b:
        raise HTTPException(400, "Both images required")
    try:
        return compare_pair_from_bytes(data_a, data_b)
    except Exception as e:
        raise HTTPException(400, f"Comparison failed: {e}") from e


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
