"""Align FER2013 faces with MTCNN when available, otherwise OpenCV."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import FER2013_DIR, FER2013_RAW_DIR
from face_detector import FaceDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FER2013_RAW_DIR)
    parser.add_argument("--output", type=Path, default=FER2013_DIR)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--detector", choices=["auto", "yunet", "mtcnn", "haar"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def align_face(image: np.ndarray, detector: FaceDetector, output_size: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Detection on a larger image is more reliable for the original 48x48 files.
    scale = 2.0
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    detections = detector.detect(scaled)
    if not detections:
        return cv2.resize(image, (output_size, output_size))

    detection = detections[0]
    if detection.keypoints:
        left_eye = np.asarray(detection.keypoints["left_eye"], dtype=np.float32)
        right_eye = np.asarray(detection.keypoints["right_eye"], dtype=np.float32)
        eye_center = (left_eye + right_eye) / 2.0
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        matrix = cv2.getRotationMatrix2D((float(eye_center[0]), float(eye_center[1])), float(angle), 1.0)
        segmented = cv2.warpAffine(
            scaled,
            matrix,
            (scaled.shape[1], scaled.shape[0]),
            flags=cv2.INTER_CUBIC,
        )
        crop = segmented[
            max(0, detection.y1) : min(scaled.shape[0], detection.y2),
            max(0, detection.x1) : min(scaled.shape[1], detection.x2),
        ]
    else:
        crop = detector.crop(scaled, detection, output_size=output_size, margin=0.12)

    if crop.size == 0:
        return cv2.resize(image, (output_size, output_size))
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input dataset directory not found: {args.input}")
    detector = FaceDetector(backend=args.detector, min_face_size=16)
    print(f"Detector: {detector.backend}")

    total_saved = 0
    total_skipped = 0
    for split in ("train", "val", "test"):
        for emotion in ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"):
            source_dir = args.input / split / emotion
            if not source_dir.is_dir():
                continue
            target_dir = args.output / split / emotion
            target_dir.mkdir(parents=True, exist_ok=True)
            image_paths = sorted(
                path for path in source_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for source_path in tqdm(image_paths, desc=f"{split}/{emotion}", leave=False):
                target_path = target_dir / source_path.name
                if target_path.exists() and not args.overwrite:
                    total_skipped += 1
                    continue
                image = cv2.imread(str(source_path))
                if image is None:
                    continue
                aligned = align_face(image, detector, args.size)
                cv2.imwrite(str(target_path), aligned)
                total_saved += 1

    print(f"Done. saved={total_saved}, skipped={total_skipped}, output={args.output}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()

