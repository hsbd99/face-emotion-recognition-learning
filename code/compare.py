"""Compare the CNN and HOG + SVM baselines from saved metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_legacy_text(path: Path) -> dict | None:
    if not path.is_file():
        return None
    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip().replace("%", "")
            if key == "accuracy":
                values["accuracy"] = float(value) / 100.0
            elif key == "f1-score":
                values["f1_weighted"] = float(value)
    return values or None


def main() -> None:
    args = parse_args()
    cnn_metrics = read_json(args.output_dir / "metrics.json")
    hog_metrics = read_json(args.output_dir / "hog_svm_results.json")
    if cnn_metrics is None:
        cnn_metrics = read_legacy_text(args.output_dir / "deep_results.txt")
    if hog_metrics is None:
        hog_metrics = read_legacy_text(args.output_dir / "hog_svm_results.txt")

    if cnn_metrics is None or hog_metrics is None:
        raise FileNotFoundError(
            "Required metric files are missing. Run judge.py and HOGSVM.py first."
        )

    cnn_fer = cnn_metrics.get("fer2013", cnn_metrics)
    cnn_accuracy = float(cnn_fer["accuracy"])
    cnn_f1 = float(cnn_fer.get("f1_weighted", cnn_fer.get("f1-score", 0.0)))
    hog_accuracy = float(hog_metrics["accuracy"])
    hog_f1 = float(hog_metrics.get("f1_weighted", hog_metrics.get("f1-score", 0.0)))

    print("=" * 48)
    print("Method comparison")
    print("=" * 48)
    print(f"{'Method':<16} {'Accuracy':>10} {'Weighted F1':>14}")
    print("-" * 48)
    print(f"{'HOG + SVM':<16} {hog_accuracy * 100:>9.2f}% {hog_f1:>14.4f}")
    print(f"{'CNN':<16} {cnn_accuracy * 100:>9.2f}% {cnn_f1:>14.4f}")
    print("-" * 48)
    print(f"{'Improvement':<16} {(cnn_accuracy - hog_accuracy) * 100:>+9.2f}% {cnn_f1 - hog_f1:>+14.4f}")

    output = {
        "hog_svm": {"accuracy": hog_accuracy, "f1_weighted": hog_f1},
        "cnn": {"accuracy": cnn_accuracy, "f1_weighted": cnn_f1},
        "improvement": {
            "accuracy": cnn_accuracy - hog_accuracy,
            "f1_weighted": cnn_f1 - hog_f1,
        },
    }
    with open(args.output_dir / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
