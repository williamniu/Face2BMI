# Face-to-BMI Recreation

Reimplementation of **Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media** by Kocabey et al. (ICWSM 2017), using the provided `bmi_data` dataset.

This repository contains the full final-project package: source code, trained model, generated evaluation reports, figures, write-up, and a local FastAPI demo website with upload and webcam capture.

## Quick Start

Clone or open this repository, then run from the repository root:

```bash
source .venv/bin/activate
cd web
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open one of these URLs in a normal browser:

```text
http://127.0.0.1:8000
http://localhost:8000
```

Use a normal Chrome/Edge browser for webcam capture. Embedded IDE browsers may block camera permissions.

To stop the server, press `Ctrl+C` in the terminal running `uvicorn`.

If the repository was cloned without model artifacts, follow the setup and reproduction steps below before running the demo.

## What the Demo Does

The web demo supports:

- Uploading one face image and predicting BMI.
- Starting the webcam, aligning the face inside a square `224 x 224` model-input guide, capturing a frame, and predicting BMI.
- Uploading two face images and predicting which one has higher BMI.
- Loading sample test-set images for quick demonstration.

The API is implemented in `web/app.py` and exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Demo website |
| `/api/health` | GET | Server/model health check |
| `/api/samples` | GET | Sample test-set images |
| `/api/sample-image/{filename}` | GET | Serve a sample image |
| `/api/predict` | POST | Predict BMI for one uploaded image |
| `/api/compare` | POST | Compare two uploaded images |

## Current Results

The final tuned model uses VGG16 ImageNet `fc6`-style embeddings plus epsilon-SVR.

| Metric | Value |
|---|---:|
| Overall Pearson r | 0.409 |
| Male Pearson r | 0.485 |
| Female Pearson r | 0.292 |
| MAE | 6.299 |
| RMSE | 8.722 |
| Pairwise accuracy | 0.636 |

The paper reports stronger performance with VGG-Face features:

| Method | Overall r | Male r | Female r |
|---|---:|---:|---:|
| Paper VGG-Net | 0.47 | 0.58 | 0.36 |
| Paper VGG-Face | 0.65 | 0.71 | 0.57 |
| This reproduction | 0.409 | 0.485 | 0.292 |

The most important remaining performance gap is the feature extractor: the original paper's best result uses VGG-Face, while this implementation uses `torchvision` VGG16 pretrained on ImageNet.

## Project Layout

```text
.
├── bmi_data/
│   ├── data.csv
│   └── Images/
├── models/
│   ├── embeddings/
│   │   ├── train_vgg16_fc6.npz
│   │   └── test_vgg16_fc6.npz
│   ├── face2bmi_svr.joblib
│   └── training_metadata.json
├── reports/
│   ├── figures/
│   ├── manifests/
│   ├── dataset_audit.json
│   ├── evaluation_report.json
│   ├── evaluation_summary.md
│   └── svr_candidate_results.json
├── scripts/
│   ├── audit_data.py
│   ├── train_model.py
│   └── evaluate_model.py
├── src/face2bmi/
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── inference.py
│   └── train.py
├── web/
│   ├── app.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── implementation_report.md
├── paper.md
├── requirements.txt
└── README.md
```

## Repository Artifact Notes

This project can be used as a standalone GitHub repository. Do not commit the virtual environment:

```text
.venv/
```

The trained model and cached embeddings are generated artifacts:

```text
models/face2bmi_svr.joblib
models/embeddings/*.npz
```

Include them if the repo must run the demo immediately after cloning. Exclude them if the repo should stay lightweight; in that case, users can regenerate them with `python scripts/train_model.py --force-features --n-jobs 1`.

The dataset folder is required for full retraining and sample-image demo behavior:

```text
bmi_data/data.csv
bmi_data/Images/
```

Before publishing publicly, confirm that sharing the provided image data is allowed by the course/data license.

## Reproducible Setup

Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already have a working `.venv`, you can activate it directly instead of recreating it.

## Reproduce the Pipeline

Run these commands from the repository root with the virtual environment activated.

### 1. Audit Dataset

```bash
python scripts/audit_data.py
```

Outputs:

- `reports/dataset_audit.json`
- `reports/manifests/full_manifest.csv`
- `reports/manifests/train_manifest.csv`
- `reports/manifests/test_manifest.csv`
- `reports/manifests/missing_images.csv`

### 2. Train Model

```bash
python scripts/train_model.py --n-jobs 1
```

This trains the SVR model using cached embeddings when available. If embeddings are missing or need to be regenerated:

```bash
python scripts/train_model.py --force-features --n-jobs 1
```

Use more CPU workers only when desired:

```bash
python scripts/train_model.py --n-jobs 4
```

Outputs:

- `models/embeddings/*.npz`
- `models/face2bmi_svr.joblib`
- `models/training_metadata.json`

### 3. Evaluate Model

```bash
python scripts/evaluate_model.py
```

Outputs:

- `reports/evaluation_report.json`
- `reports/evaluation_summary.md`

### 4. Run Demo Website

```bash
cd web
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Dataset Notes

The CSV matches the paper's reported 4,206 examples, but not every referenced image is present locally.

| Metric | Count |
|---|---:|
| CSV rows | 4,206 |
| Available images | 3,962 |
| Missing images | 244 |
| Available train images | 3,210 |
| Available test images | 752 |

No dataset rows or labels were altered. Rows with missing image files are excluded from training and evaluation and listed in `reports/manifests/missing_images.csv`.

## Report and Figures

The main write-up is:

```text
implementation_report.md
```

Generated report figures are in:

```text
reports/figures/
```

Important figures include:

- `paper_metric_comparison.png`
- `training_testing_log_summary.png`
- `bmi_category_distribution.png`
- `svr_tuning_results.png`
- `predicted_vs_actual.png`
- `residuals_by_bmi.png`
- `pairwise_accuracy_by_bin.png`

These figures are suitable as a base for the final presentation.

## Model Details

Feature extraction:

- Pretrained `torchvision.models.vgg16`
- ImageNet weights
- `fc6`-style 4096-dimensional embedding
- Frozen feature extractor

Regression:

- `StandardScaler`
- `SVR(kernel="rbf")`
- Tuned parameters:
  - `C = 30`
  - `epsilon = 0.01`
  - `gamma = 1e-5`

The final model file is:

```text
models/face2bmi_svr.joblib
```

## Ethics and Limitations

This is an educational final project. BMI estimates from face images are noisy and should not be used for medical, employment, insurance, legal, or personal judgment decisions.

Known limitations:

- Uses ImageNet VGG16 instead of the paper's VGG-Face model.
- 244 referenced images are missing from the provided dataset.
- Individual predictions are noisy.
- Very high BMI examples tend to be underpredicted.
- Female subset performance is lower than male subset performance.
- Race-bias analysis cannot be reproduced because race labels are not provided.

## Best Next Improvement

The most paper-faithful next step is to replace ImageNet VGG16 features with actual VGG-Face `fc6` features and match the original preprocessing as closely as possible. The paper's own comparison suggests this is the main path from approximately VGG-Net-level performance toward the reported VGG-Face result.
