from pathlib import Path

FINAL_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = FINAL_ROOT / "bmi_data" / "data.csv"
IMAGES_DIR = FINAL_ROOT / "bmi_data" / "Images"
REPORTS_DIR = FINAL_ROOT / "reports"
MODELS_DIR = FINAL_ROOT / "models"
MANIFESTS_DIR = REPORTS_DIR / "manifests"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
ALIGNED_DIR = FINAL_ROOT / "bmi_data" / "aligned"

# ImageNet normalization (VGG-style)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

# FaceNet (facenet-pytorch InceptionResnetV1 / VGGFace2) preprocessing
FACENET_INPUT_SIZE = 160
FACENET_MEAN = (0.5, 0.5, 0.5)
FACENET_STD = (0.5, 0.5, 0.5)

# Backbone registry. Each entry describes how to build the feature extractor.
# Used by features.py to pick the right model + preprocessing.
BACKBONES = {
    "vgg16_imagenet": {
        "feature_dim": 4096,
        "input_size": INPUT_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "cache_name": "vgg16_fc6",
    },
    "facenet_vggface2": {
        "feature_dim": 512,
        "input_size": FACENET_INPUT_SIZE,
        "mean": FACENET_MEAN,
        "std": FACENET_STD,
        "cache_name": "facenet_vggface2",
    },
    "facenet_casia": {
        "feature_dim": 512,
        "input_size": FACENET_INPUT_SIZE,
        "mean": FACENET_MEAN,
        "std": FACENET_STD,
        "cache_name": "facenet_casia",
    },
}

# Default backbone for the improved system (paper: VGG-Face → here: VGGFace2-trained InceptionResnetV1).
DEFAULT_BACKBONE = "facenet_vggface2"

# Backbones combined into the final ensemble.
# Two face-trained backbones (VGGFace2 and CASIA-Webface InceptionResnetV1)
# carry the performance; vgg16_imagenet is a weak but uncorrelated diverse
# learner whose predictions still nudge the averaged ensemble upward.
ENSEMBLE_BACKBONES = ["facenet_vggface2", "facenet_casia", "vgg16_imagenet"]
# All trained backbones go into the deployed prediction ensemble.
DEPLOY_BACKBONES = ENSEMBLE_BACKBONES

# BMI category boundaries from paper (Section 5.3)
BMI_CATEGORIES = [
    ("Underweight", 16.0, 18.5),
    ("Normal", 18.5, 25.0),
    ("Overweight", 25.0, 30.0),
    ("Moderately obese", 30.0, 35.0),
    ("Severely obese", 35.0, 40.0),
    ("Very severely obese", 40.0, float("inf")),
]
