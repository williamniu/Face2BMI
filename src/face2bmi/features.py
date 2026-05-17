"""Deep feature extraction (VGG16 fc6-style embeddings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from face2bmi.config import EMBEDDINGS_DIR, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def get_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FaceImageDataset(Dataset):
    def __init__(self, image_paths: list[str | Path], transform=None):
        self.paths = [Path(p) for p in image_paths]
        self.transform = transform or get_preprocess()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, str(path)


class VGGFeatureExtractor(nn.Module):
    """Extract 4096-d fc6 features from VGG16 (ImageNet), mirroring paper setup."""

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights)
        # features + avgpool + first FC (fc6)
        self.backbone = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            vgg.classifier[0],  # Linear 25088 -> 4096
            nn.ReLU(inplace=True),
        )
        self.backbone.eval()
        self.backbone.to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_batch(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)
        out = self.forward(x)
        return out.cpu().numpy()


def extract_embeddings(
    image_paths: list[str | Path],
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> np.ndarray:
    dataset = FaceImageDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    extractor = VGGFeatureExtractor(device=device)
    chunks: list[np.ndarray] = []
    for batch, _ in loader:
        chunks.append(extractor.extract_batch(batch))
    return np.vstack(chunks)


def cache_split_embeddings(
    manifest: "pd.DataFrame",
    split_name: str,
    force: bool = False,
) -> Path:
    import pandas as pd

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBEDDINGS_DIR / f"{split_name}_vgg16_fc6.npz"
    if out_path.exists() and not force:
        return out_path

    paths = manifest["image_path"].tolist()
    embeddings = extract_embeddings(paths)
    np.savez(
        out_path,
        embeddings=embeddings,
        bmi=manifest["bmi"].values.astype(np.float32),
        gender=manifest["gender"].values,
        names=manifest["name"].values,
        paths=np.array(paths),
    )
    return out_path


def load_cached_embeddings(split_name: str) -> dict:
    path = EMBEDDINGS_DIR / f"{split_name}_vgg16_fc6.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {path}. Run training/feature extraction first."
        )
    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "bmi": data["bmi"],
        "gender": data["gender"],
        "names": data["names"],
        "paths": data["paths"],
    }
