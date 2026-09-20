"""Evaluate a trained checkpoint on FER2013 and optionally CK+."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CKPLUS_DIR, EMOTIONS, OUTPUT_DIR
from data import CKPlusDataset, FERDataset, get_transforms, resolve_fer_root
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="FER2013 aligned/raw root")
    parser.add_argument("--ckplus-dir", type=Path, default=CKPLUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dataset", choices=["fer", "ck+", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def evaluate_dataset(model, dataset, device: torch.device, batch_size: int, workers: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="evaluate", leave=False):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())
            probabilities.append(probs)

    metrics = {
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(EMOTIONS))),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(EMOTIONS)))).tolist(),
        "_predictions": predictions,
        "_labels": labels,
        "_probabilities": np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, len(EMOTIONS))),
    }
    return metrics


def save_confusion_matrix(matrix: list[list[int]], path: Path, title: str) -> None:
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(EMOTIONS)),
        yticks=np.arange(len(EMOTIONS)),
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = values.max() / 2.0 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def printable_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}:")
    print(f"  samples: {metrics['samples']}")
    print(f"  accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"  macro F1: {metrics['f1_macro']:.4f}")
    print(f"  macro precision: {metrics['precision_macro']:.4f}")
    print(f"  macro recall: {metrics['recall_macro']:.4f}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = [
            Path(__file__).resolve().parent / "emotion_best.pth",
            Path(__file__).resolve().parent.parent / "exp_result" / "checkpoints" / "emotion_best.pth",
        ]
        checkpoint_path = next((path for path in candidates if path.is_file()), candidates[-1])
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    image_size = int(metadata.get("image_size", 96))
    transform = get_transforms(image_size, train=False)

    results: dict[str, dict] = {}
    fer_root = resolve_fer_root(args.data_dir)
    if args.dataset in {"fer", "both"}:
        fer_dataset = FERDataset(fer_root, split="test", transform=transform)
        fer_metrics = evaluate_dataset(model, fer_dataset, device, args.batch_size, args.workers)
        results["fer2013"] = fer_metrics
        save_confusion_matrix(
            fer_metrics["confusion_matrix"],
            args.output_dir / "confusion_matrix_fer2013.png",
            "FER2013 confusion matrix",
        )

    if args.dataset in {"ck+", "both"} and Path(args.ckplus_dir).is_dir():
        ck_dataset = CKPlusDataset(args.ckplus_dir, transform=transform)
        ck_metrics = evaluate_dataset(model, ck_dataset, device, args.batch_size, args.workers)
        results["ckplus"] = ck_metrics
        save_confusion_matrix(
            ck_metrics["confusion_matrix"],
            args.output_dir / "confusion_matrix_ckplus.png",
            "CK+ confusion matrix",
        )

    clean_results = {}
    for name, metrics in results.items():
        clean = {key: value for key, value in metrics.items() if not key.startswith("_")}
        clean_results[name] = clean
        printable_metrics(name.upper(), metrics)

    output_path = args.output_dir / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(clean_results, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
