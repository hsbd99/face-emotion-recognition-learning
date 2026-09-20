"""Train the HOG + SVM baseline with reproducible metrics output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import SVC
from tqdm import tqdm

from config import EMOTIONS, EMOTION_TO_INDEX, OUTPUT_DIR
from data import resolve_fer_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--c", type=float, default=10.0)
    return parser.parse_args()


def extract_hog(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9)
    return descriptor.compute(gray).flatten()


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for emotion in EMOTIONS:
        class_dir = root / split / emotion
        if not class_dir.is_dir():
            continue
        for path in tqdm(sorted(class_dir.iterdir()), desc=f"{split}/{emotion}", leave=False):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            features.append(extract_hog(image))
            labels.append(EMOTION_TO_INDEX[emotion])
    if not features:
        raise RuntimeError(f"No usable images found for split '{split}' under {root}")
    return np.stack(features), np.asarray(labels)


def main() -> None:
    args = parse_args()
    root = resolve_fer_root(args.data_dir)
    print("Extracting HOG features...")
    train_features, train_labels = load_split(root, "train")
    test_features, test_labels = load_split(root, "test")
    print(f"Train: {len(train_features)} | Test: {len(test_features)}")

    classifier = SVC(kernel="rbf", C=args.c, gamma="scale", class_weight="balanced")
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    metrics = {
        "samples": int(len(test_labels)),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "f1_weighted": float(f1_score(test_labels, predictions, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=list(range(len(EMOTIONS))),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hog_svm.joblib"
    joblib.dump({"model": classifier, "class_names": EMOTIONS, "image_size": 48}, model_path)
    with open(args.output_dir / "hog_svm_results.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with open(args.output_dir / "hog_svm_results.txt", "w", encoding="utf-8") as handle:
        handle.write(f"Accuracy: {metrics['accuracy'] * 100:.2f}%\n")
        handle.write(f"F1-score: {metrics['f1_weighted']:.4f}\n")

    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"Saved model and metrics under {args.output_dir}")


if __name__ == "__main__":
    main()
