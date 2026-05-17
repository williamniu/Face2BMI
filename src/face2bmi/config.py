from pathlib import Path

FINAL_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = FINAL_ROOT / "bmi_data" / "data.csv"
IMAGES_DIR = FINAL_ROOT / "bmi_data" / "Images"
REPORTS_DIR = FINAL_ROOT / "reports"
MODELS_DIR = FINAL_ROOT / "models"
MANIFESTS_DIR = REPORTS_DIR / "manifests"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

# ImageNet normalization (VGG-style)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

# BMI category boundaries from paper (Section 5.3)
BMI_CATEGORIES = [
    ("Underweight", 16.0, 18.5),
    ("Normal", 18.5, 25.0),
    ("Overweight", 25.0, 30.0),
    ("Moderately obese", 30.0, 35.0),
    ("Severely obese", 35.0, 40.0),
    ("Very severely obese", 40.0, float("inf")),
]
