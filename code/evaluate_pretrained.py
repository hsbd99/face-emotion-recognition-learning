"""Evaluate the OpenCV Zoo pretrained FER model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from config import CKPLUS_DIR, EMOTIONS, OUTPUT_DIR
from data import CKPlusDataset, FERDataset, resolve_fer_root
from judge import printable_metrics, save_confusion_matrix
from pretrained_fer import DEFAULT_MODEL_PATH, OpenCVFacialExpressionRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--ckplus-dir", type=Path, default=CKPLUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dataset", choices=["fer", "ck+", "both"], default="both")
    return parser.parse_args()


def evaluate(dataset, recognizer) -> dict:
    labels: list[int] = []
    predictions: list[int] = []
    for path, label in tqdm(dataset.samples, desc="evaluate", leave=False):
        image = cv2.imread(str(path))
        if image is None:
            continue
        probabilities = recognizer.infer_probs(image)
        labels.append(int(label))
        predictions.append(int(np.argmax(probabilities)))

    return {
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
    }


def main() -> None:
    args = parse_args()
    recognizer = OpenCVFacialExpressionRecognizer(args.model)
    results: dict[str, dict] = {}

    if args.dataset in {"fer", "both"}:
        results["fer2013"] = evaluate(
            FERDataset(resolve_fer_root(args.data_dir), split="test", transform=None),
            recognizer,
        )
        save_confusion_matrix(
            results["fer2013"]["confusion_matrix"],
            args.output_dir / "confusion_matrix_fer2013_opencv.png",
            "FER2013 confusion matrix (OpenCV Zoo)",
        )

    if args.dataset in {"ck+", "both"} and Path(args.ckplus_dir).is_dir():
        results["ckplus"] = evaluate(CKPlusDataset(args.ckplus_dir, transform=None), recognizer)
        save_confusion_matrix(
            results["ckplus"]["confusion_matrix"],
            args.output_dir / "confusion_matrix_ckplus_opencv.png",
            "CK+ confusion matrix (OpenCV Zoo)",
        )

    for name, metrics in results.items():
        printable_metrics(f"{name.upper()} (OpenCV Zoo)", metrics)

    output_path = args.output_dir / "metrics_opencv.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
