"""Dataset and data-loading utilities for FER2013 and CK+."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import CKPLUS_ALIASES, EMOTIONS, EMOTION_TO_INDEX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Return deterministic evaluation transforms or light training augmentation."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.80, 1.0),
                    ratio=(0.90, 1.10),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, fill=0),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class FERDataset(Dataset):
    """Directory-based FER2013 dataset with a fixed seven-class label order."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: transforms.Compose | None = None,
        strict: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.class_to_index = dict(EMOTION_TO_INDEX)
        self.samples: list[tuple[Path, int]] = []

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"FER2013 split directory not found: {split_dir}")

        for emotion in EMOTIONS:
            class_dir = split_dir / emotion
            images = _list_images(class_dir)
            if strict and not images:
                raise FileNotFoundError(f"No images found for class '{emotion}': {class_dir}")
            self.samples.extend((path, self.class_to_index[emotion]) for path in images)

        if not self.samples:
            raise RuntimeError(f"No samples found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label

    def class_counts(self) -> dict[str, int]:
        counts = Counter(label for _, label in self.samples)
        return {emotion: counts.get(EMOTION_TO_INDEX[emotion], 0) for emotion in EMOTIONS}


class CKPlusDataset(Dataset):
    """CK+ dataset with aliases mapped to the FER2013 seven-class space."""

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        strict: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"CK+ directory not found: {self.root}")

        for source_name, target_name in CKPLUS_ALIASES.items():
            class_dir = self.root / source_name
            images = _list_images(class_dir)
            if strict and not images:
                raise FileNotFoundError(f"No images found for CK+ class '{source_name}'")
            self.samples.extend((path, EMOTION_TO_INDEX[target_name]) for path in images)

        if not self.samples:
            raise RuntimeError(f"No CK+ images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """Return inverse-frequency weights for the seven FER2013 classes."""
    labels = torch.tensor([int(label) for _, label in dataset], dtype=torch.long)
    counts = torch.bincount(labels, minlength=len(EMOTIONS)).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * len(EMOTIONS))
    return weights


def create_weighted_sampler(
    dataset: Dataset,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Oversample minority classes without duplicating files on disk."""
    labels = [int(label) for _, label in dataset]
    counts = Counter(labels)
    sample_weights = [1.0 / counts[int(label)] for _, label in dataset]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples or len(dataset),
        replacement=True,
    )


def create_dataloaders(
    dataset_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    balanced: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Build training and validation loaders from an aligned FER2013 root."""
    train_dataset = FERDataset(
        dataset_root,
        split="train",
        transform=get_transforms(image_size, train=True),
    )
    val_dataset = FERDataset(
        dataset_root,
        split="val",
        transform=get_transforms(image_size, train=False),
    )

    sampler = create_weighted_sampler(train_dataset) if balanced else None
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_dataset.class_counts()


def resolve_fer_root(preferred: str | Path | None = None) -> Path:
    """Resolve the aligned dataset first, then fall back to the raw layout."""
    if preferred is not None:
        path = Path(preferred).expanduser().resolve()
        if path.is_dir():
            return path
        raise FileNotFoundError(f"FER2013 directory not found: {path}")

    from config import FER2013_DIR, FER2013_RAW_DIR

    for candidate in (FER2013_DIR, FER2013_RAW_DIR):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Neither aligned nor raw FER2013 directory exists: {FER2013_DIR}, {FER2013_RAW_DIR}"
    )


__all__ = [
    "CKPlusDataset",
    "FERDataset",
    "compute_class_weights",
    "create_dataloaders",
    "create_weighted_sampler",
    "get_transforms",
    "resolve_fer_root",
]
