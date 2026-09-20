"""Train a seven-class facial-expression classifier."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import CHECKPOINT_DIR, DEFAULT_ARCH, DEFAULT_IMAGE_SIZE, EMOTIONS, OUTPUT_DIR
from data import create_dataloaders, resolve_fer_root
from model import build_model, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="FER2013 aligned/raw root")
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        choices=["mobilenet_v3_small", "resnet18", "simple_cnn"],
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4, help="Classifier learning rate")
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-balanced", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    if not hasattr(model, "features"):
        return
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("features")]
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("features")]
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, leave=False, desc="train" if training else "val")

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.4f}")

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 12)))
    device = resolve_device(args.device)
    data_root = resolve_fer_root(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_counts = create_dataloaders(
        data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        balanced=not args.no_balanced,
    )

    model = build_model(
        arch=args.arch,
        num_classes=len(EMOTIONS),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    config = vars(args).copy()
    config["data_root"] = str(data_root)
    config["device"] = str(device)
    config["class_counts"] = class_counts
    config["parameters"] = model.count_parameters()
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    epochs_without_improvement = 0
    best_path = args.output_dir / "emotion_best.pth"
    last_path = args.output_dir / "emotion_last.pth"

    print(f"Device: {device}")
    print(f"Architecture: {args.arch} ({model.count_parameters():,} trainable parameters)")
    print(f"Dataset: {data_root}")
    print(f"Class counts: {class_counts}")

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        trainable = not (args.freeze_epochs > 0 and epoch <= args.freeze_epochs)
        set_backbone_trainable(model, trainable)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        elapsed = time.perf_counter() - start
        current_lr = optimizer.param_groups[-1]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": current_lr,
            "elapsed_seconds": elapsed,
            "backbone_trainable": trainable,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s"
        )

        save_checkpoint(
            last_path,
            model,
            image_size=args.image_size,
            epoch=epoch,
            metrics={"val_accuracy": val_acc, "val_loss": val_loss},
            optimizer=optimizer,
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                image_size=args.image_size,
                epoch=epoch,
                metrics={
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "train_loss": train_loss,
                },
            )
            print(f"  saved best checkpoint: {best_path} (val acc {val_acc:.4f})")
        else:
            epochs_without_improvement += 1

        with open(args.output_dir / "training_history.json", "w", encoding="utf-8") as handle:
            json.dump({"config": config, "history": history, "best_accuracy": best_accuracy}, handle, indent=2, default=str)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break

    print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

