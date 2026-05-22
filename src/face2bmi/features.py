"""Deep feature extraction with pluggable backbones.

Supports:
  - vgg16_imagenet : torchvision VGG16 fc6 (paper-faithful VGG-Net baseline)
  - facenet_vggface2: facenet-pytorch InceptionResnetV1 pretrained on VGGFace2
  - facenet_casia  : facenet-pytorch InceptionResnetV1 pretrained on CASIA-Webface
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from face2bmi.config import (
    BACKBONES,
    DEFAULT_BACKBONE,
    EMBEDDINGS_DIR,
)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_preprocess(backbone: str) -> Callable:
    cfg = BACKBONES[backbone]
    return transforms.Compose(
        [
            transforms.Resize((cfg["input_size"], cfg["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )


class FaceImageDataset(Dataset):
    def __init__(self, image_paths: list[str | Path], transform: Callable):
        self.paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, str(path)


class _VGG16FC6(nn.Module):
    """ImageNet VGG16 fc6-style 4096-d embedding (paper-faithful VGG-Net path)."""

    def __init__(self, device: torch.device):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights)
        self.backbone = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            vgg.classifier[0],
            nn.ReLU(inplace=True),
        )
        self.backbone.eval()
        self.backbone.to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _FaceNet(nn.Module):
    """InceptionResnetV1 (facenet-pytorch) face embedding — 512-d."""

    def __init__(self, device: torch.device, pretrained: str):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self.backbone = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_backbone(name: str, device: torch.device | None = None) -> nn.Module:
    """Construct a frozen feature extractor by name."""
    device = device or _select_device()
    if name == "vgg16_imagenet":
        return _VGG16FC6(device)
    if name == "facenet_vggface2":
        return _FaceNet(device, pretrained="vggface2")
    if name == "facenet_casia":
        return _FaceNet(device, pretrained="casia-webface")
    raise ValueError(f"Unknown backbone: {name}")


@torch.inference_mode()
def _extract(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for batch, _ in loader:
        out = model(batch.to(device))
        chunks.append(out.detach().cpu().numpy())
    return np.vstack(chunks)


def extract_embeddings(
    image_paths: list[str | Path],
    backbone: str = DEFAULT_BACKBONE,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or _select_device()
    transform = get_preprocess(backbone)
    dataset = FaceImageDataset(image_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_backbone(backbone, device=device)
    return _extract(model, loader, device)


def _cache_path(backbone: str, split_name: str) -> Path:
    cache_name = BACKBONES[backbone]["cache_name"]
    return EMBEDDINGS_DIR / f"{split_name}_{cache_name}.npz"


def cache_split_embeddings(
    manifest,
    split_name: str,
    backbone: str = DEFAULT_BACKBONE,
    force: bool = False,
    batch_size: int = 32,
) -> Path:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _cache_path(backbone, split_name)
    if out_path.exists() and not force:
        return out_path

    paths = manifest["image_path"].tolist()
    embeddings = extract_embeddings(paths, backbone=backbone, batch_size=batch_size)
    np.savez(
        out_path,
        embeddings=embeddings,
        bmi=manifest["bmi"].values.astype(np.float32),
        gender=manifest["gender"].values,
        names=manifest["name"].values,
        paths=np.array(paths),
        backbone=np.array(backbone),
    )
    return out_path


def load_cached_embeddings(split_name: str, backbone: str = DEFAULT_BACKBONE) -> dict:
    path = _cache_path(backbone, split_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {path}. Run feature extraction first."
        )
    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "bmi": data["bmi"],
        "gender": data["gender"],
        "names": data["names"],
        "paths": data["paths"],
    }
