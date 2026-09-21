"""Model factory, checkpoint helpers and a lightweight Grad-CAM target."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

from config import EMOTIONS


class EmotionNet(nn.Module):
    """Transfer-learning classifier with a stable feature-map interface."""

    def __init__(
        self,
        num_classes: int = 7,
        arch: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        if arch == "mobilenet_v3_small":
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                backbone = models.mobilenet_v3_small(weights=weights)
            except Exception as exc:  # pragma: no cover - depends on network/cache
                if not pretrained:
                    raise
                print(f"Warning: pretrained weights unavailable ({exc}); using random weights.")
                backbone = models.mobilenet_v3_small(weights=None)

            in_features = backbone.classifier[0].in_features
            self.features = backbone.features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        elif arch == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                backbone = models.resnet18(weights=weights)
            except Exception as exc:  # pragma: no cover - depends on network/cache
                if not pretrained:
                    raise
                print(f"Warning: pretrained weights unavailable ({exc}); using random weights.")
                backbone = models.resnet18(weights=None)

            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            in_features = backbone.fc.in_features
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        elif arch == "fer_cnn":
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        elif arch == "simple_cnn":
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feature_map = self.forward_features(x)
        pooled = self.avgpool(feature_map).flatten(1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, feature_map
        return logits

    def get_gradcam_layer(self) -> nn.Module:
        """Return the last convolutional layer used by Grad-CAM."""
        if self.arch == "simple_cnn":
            return self.features[-3]
        if self.arch == "fer_cnn":
            return self.features[-1]
        if self.arch == "mobilenet_v3_small":
            return self.features[-1]
        if self.arch == "resnet18":
            return self.features[-1][-1].conv2
        raise ValueError(f"No Grad-CAM layer configured for {self.arch}")

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_model(
    arch: str = "mobilenet_v3_small",
    num_classes: int = len(EMOTIONS),
    pretrained: bool = True,
    dropout: float = 0.2,
) -> EmotionNet:
    return EmotionNet(
        num_classes=num_classes,
        arch=arch,
        pretrained=pretrained,
        dropout=dropout,
    )


def save_checkpoint(
    path: str | Path,
    model: EmotionNet,
    *,
    image_size: int,
    epoch: int,
    metrics: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 2,
        "arch": model.arch,
        "num_classes": model.num_classes,
        "class_names": EMOTIONS,
        "image_size": image_size,
        "epoch": epoch,
        "metrics": metrics or {},
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    pretrained: bool = False,
) -> tuple[EmotionNet, dict[str, Any]]:
    """Load a versioned checkpoint or the original raw state_dict."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        arch = checkpoint.get("arch", "mobilenet_v3_small")
        num_classes = int(checkpoint.get("num_classes", len(EMOTIONS)))
        state_dict = checkpoint["model_state"]
        metadata = checkpoint
    else:
        # Original project checkpoints were raw SimpleCNN/EmotionNet state dicts.
        arch = "simple_cnn"
        num_classes = len(EMOTIONS)
        state_dict = checkpoint
        metadata = {
            "format_version": 1,
            "arch": arch,
            "num_classes": num_classes,
            "image_size": 48,
            "class_names": EMOTIONS,
            "metrics": {},
        }

    model = build_model(arch=arch, num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata


if __name__ == "__main__":
    model = build_model(pretrained=False)
    dummy = torch.randn(2, 3, 96, 96)
    output = model(dummy)
    print("output:", output.shape, "parameters:", model.count_parameters())


