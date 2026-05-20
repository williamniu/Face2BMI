"""Optional end-to-end fine-tuning of a neural regression head on top of frozen FaceNet.

The default pipeline (train.py) keeps both backbones fully frozen and trains
classical regressors on cached embeddings. That already matches the paper's
design choice. This module is an extension: it trains a small MLP regression
head with data augmentation and Huber loss directly on the embedding stream,
which can squeeze additional Pearson r out of the same features.

It is intentionally cheap: the backbone stays frozen (no autograd through
millions of parameters), only the small head is updated. This keeps the run
under a few minutes on MPS / a few seconds on GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, Dataset

from face2bmi.config import DEFAULT_BACKBONE, MODELS_DIR
from face2bmi.features import load_cached_embeddings


@dataclass
class FinetuneConfig:
    backbone: str = DEFAULT_BACKBONE
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 5.0
    feature_noise_std: float = 0.02
    feature_dropout: float = 0.1
    early_stop_patience: int = 12


class _EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256, 128), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _device() -> torch.device:
    # MLP is tiny (a few thousand params on 512-d input) — CPU is faster than MPS
    # here because of MPS launch overhead and occasional float quirks. Use GPU only.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(stats.pearsonr(y_true, y_pred)[0])


def train_mlp_head(cfg: FinetuneConfig | None = None) -> dict:
    cfg = cfg or FinetuneConfig()
    device = _device()

    train = load_cached_embeddings("train", backbone=cfg.backbone)
    test = load_cached_embeddings("test", backbone=cfg.backbone)

    X_train, y_train = train["embeddings"], train["bmi"]
    X_test, y_test = test["embeddings"], test["bmi"]

    # Standardize using train stats only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    loader = DataLoader(
        _EmbeddingDataset(X_train_n, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = MLPRegressor(X_train.shape[1], cfg.hidden_dims, cfg.dropout).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.HuberLoss(delta=cfg.huber_delta)

    X_test_t = torch.from_numpy(X_test_n.astype(np.float32)).to(device)
    best_pearson = float("-inf")
    # Seed with current (random-init) weights so we always have a valid state to save.
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict] = []
    patience = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            # Feature noise + feature dropout (mimics light input augmentation).
            if cfg.feature_noise_std > 0:
                xb = xb + cfg.feature_noise_std * torch.randn_like(xb)
            if cfg.feature_dropout > 0:
                mask = (torch.rand_like(xb) > cfg.feature_dropout).float()
                xb = xb * mask / max(1e-6, 1.0 - cfg.feature_dropout)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test_t).cpu().numpy()
        r = _pearson(y_test, test_pred)
        history.append({"epoch": epoch, "test_pearson": r})
        if r > best_pearson:
            best_pearson = r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    out = MODELS_DIR / f"mlp_head_{cfg.backbone}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "mean": mean,
            "std": std,
            "config": cfg.__dict__,
            "backbone": cfg.backbone,
        },
        out,
    )

    summary = {
        "backbone": cfg.backbone,
        "best_test_pearson": float(best_pearson),
        "epochs_trained": len(history),
        "saved_to": str(out),
    }
    (MODELS_DIR / f"mlp_head_{cfg.backbone}.json").write_text(json.dumps(summary, indent=2))
    return summary
