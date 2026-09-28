"""Compare saved model metrics."""
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
    values: dict[str, float] = {}
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
    cnn_all = read_json(args.output_dir / "metrics.json")
    opencv_all = read_json(args.output_dir / "metrics_opencv.json")
    hog = read_json(args.output_dir / "hog_svm_results.json") or read_legacy_text(
        args.output_dir / "hog_svm_results.txt"
    )
    cnn = (cnn_all or {}).get("fer2013", cnn_all)
    opencv = (opencv_all or {}).get("fer2013", opencv_all)

    rows = []
    if hog is not None:
        rows.append(("HOG + SVM", float(hog["accuracy"]), float(hog.get("f1_weighted", 0.0))))
    if cnn is not None:
        rows.append(("Custom CNN / ResNet", float(cnn["accuracy"]), float(cnn.get("f1_weighted", 0.0))))
    if opencv is not None:
        rows.append(("OpenCV pretrained FER", float(opencv["accuracy"]), float(opencv.get("f1_weighted", 0.0))))

    if not rows:
        raise FileNotFoundError("No metric files found. Run judge.py, evaluate_pretrained.py and HOGSVM.py first.")

    print("=" * 64)
    print("FER2013 method comparison")
    print("=" * 64)
    print(f"{'Method':<24} {'Accuracy':>12} {'Weighted F1':>16}")
    print("-" * 64)
    for name, accuracy, f1 in rows:
        print(f"{name:<24} {accuracy * 100:>11.2f}% {f1:>16.4f}")

    comparison = {
        name: {"accuracy": accuracy, "f1_weighted": f1}
        for name, accuracy, f1 in rows
    }
    with open(args.output_dir / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
