"""Run facial-expression recognition on a camera, video or image."""
from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config import DEFAULT_MODEL_PATH, EMOTIONS
from data import get_transforms
from face_detector import FaceDetection, FaceDetector
from model import load_checkpoint
from pretrained_fer import OpenCVFacialExpressionRecognizer

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 200, 255),
    "sad": (255, 100, 0),
    "surprise": (0, 255, 255),
    "neutral": (180, 180, 180),
}


class TemporalSmoother:
    def __init__(self, window: int = 6) -> None:
        self.history: dict[tuple[int, int], deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=max(1, window))
        )

    def update(self, center: tuple[int, int], probabilities: np.ndarray) -> np.ndarray:
        key = (center[0] // 48, center[1] // 48)
        self.history[key].append(probabilities.astype(np.float32))
        return np.mean(np.stack(self.history[key]), axis=0)


class TorchPredictor:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.model, metadata = load_checkpoint(checkpoint, device=device)
        self.image_size = int(metadata.get("image_size", 96))
        self.transform = get_transforms(self.image_size, train=False)
        self.device = device

    def predict(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        crop = cv2.resize(
            frame[
                max(0, detection.y1) : min(frame.shape[0], detection.y2),
                max(0, detection.x1) : min(frame.shape[1], detection.x2),
            ],
            (self.image_size, self.image_size),
        )
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()


class OpenCVPredictor:
    def __init__(self) -> None:
        self.recognizer = OpenCVFacialExpressionRecognizer()

    def predict(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        return self.recognizer.infer_probs(frame, detection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["opencv", "torch"], default="opencv")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--source", default="0", help="Camera index, video path or image path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output image/video path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector", choices=["auto", "yunet", "mtcnn", "haar"], default="auto")
    parser.add_argument("--min-face-size", type=int, default=40)
    parser.add_argument("--smooth-window", type=int, default=6)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def annotate_frame(frame, detections, predictor, smoother: TemporalSmoother):
    for detection in detections:
        probabilities = smoother.update(detection.center, predictor.predict(frame, detection))
        index = int(np.argmax(probabilities))
        label = EMOTIONS[index]
        confidence = float(probabilities[index])
        color = EMOTION_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.0%}",
            (detection.x1, max(24, detection.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def process_image(frame, detector, predictor, smoother):
    detections = detector.detect(frame)
    output = annotate_frame(frame.copy(), detections, predictor, smoother)
    return output, len(detections)


def main() -> None:
    args = parse_args()
    detector = FaceDetector(backend=args.detector, min_face_size=args.min_face_size)
    smoother = TemporalSmoother(args.smooth_window)
    device = resolve_device(args.device) if args.backend == "torch" else torch.device("cpu")
    predictor = (
        OpenCVPredictor()
        if args.backend == "opencv"
        else TorchPredictor(args.checkpoint, device)
    )
    print(f"Backend: {args.backend} | detector: {detector.backend}")

    source = parse_source(args.source)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if isinstance(source, str) and Path(source).suffix.lower() in image_suffixes:
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        output, face_count = process_image(image, detector, predictor, smoother)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), output)
        elif not args.no_display:
            cv2.imshow("Emotion Recognition", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"Processed image with {face_count} face(s).")
        return

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source {source!r}. For a camera, try --source 0.")
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    writer: cv2.VideoWriter | None = None
    fps_history: deque[float] = deque(maxlen=30)
    print("Press q or Esc to quit.")
    try:
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            output, _ = process_image(frame, detector, predictor, smoother)
            fps_history.append(1.0 / max(time.perf_counter() - started, 1e-6))
            cv2.putText(
                output,
                f"FPS: {sum(fps_history) / len(fps_history):.1f}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(args.output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        25.0,
                        (output.shape[1], output.shape[0]),
                    )
                writer.write(output)
            if not args.no_display:
                cv2.imshow("Emotion Recognition", output)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
