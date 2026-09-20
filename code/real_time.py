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
from face_detector import FaceDetector
from model import load_checkpoint

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
    """Average predictions for nearby face locations over a short window."""

    def __init__(self, window: int = 6) -> None:
        self.window = window
        self.history: dict[tuple[int, int], deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=max(1, window))
        )

    def update(self, center: tuple[int, int], probabilities: np.ndarray) -> np.ndarray:
        key = (center[0] // 48, center[1] // 48)
        self.history[key].append(probabilities.astype(np.float32))
        return np.mean(np.stack(self.history[key]), axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def predict_crop(model, transform, crop: np.ndarray, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probabilities


def annotate_frame(
    frame: np.ndarray,
    detections,
    model,
    transform,
    device,
    smoother: TemporalSmoother,
    image_size: int,
):
    for detection in detections:
        crop = detector_crop(frame, detection, image_size)
        probabilities = smoother.update(detection.center, predict_crop(model, transform, crop, device))
        index = int(np.argmax(probabilities))
        label = EMOTIONS[index]
        confidence = float(probabilities[index])
        color = EMOTION_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        text = f"{label} {confidence:.0%}"
        cv2.putText(
            frame,
            text,
            (detection.x1, max(24, detection.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def detector_crop(frame: np.ndarray, detection, image_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_x = int(detection.width * 0.08)
    pad_y = int(detection.height * 0.08)
    x1 = max(0, detection.x1 - pad_x)
    y1 = max(0, detection.y1 - pad_y)
    x2 = min(width, detection.x2 + pad_x)
    y2 = min(height, detection.y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop if crop.size else frame, (image_size, image_size))


def process_image(
    model,
    transform,
    device,
    detector,
    smoother,
    image: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, int]:
    detections = detector.detect(image)
    output = annotate_frame(image.copy(), detections, model, transform, device, smoother, image_size)
    return output, len(detections)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    image_size = int(metadata.get("image_size", 96))
    transform = get_transforms(image_size, train=False)
    detector = FaceDetector(backend=args.detector, min_face_size=args.min_face_size)
    smoother = TemporalSmoother(args.smooth_window)

    source = parse_source(args.source)
    if isinstance(source, str) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        output, face_count = process_image(
            model, transform, device, detector, smoother, image, image_size
        )
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
        raise RuntimeError(
            f"Could not open source {source!r}. For a camera, try --source 0. "
            "Check camera permissions and whether another application is using it."
        )

    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    writer: cv2.VideoWriter | None = None
    fps_history: deque[float] = deque(maxlen=30)
    print(f"Device: {device} | detector: {detector.backend} | checkpoint: {args.checkpoint}")
    print("Press q or Esc to quit.")

    try:
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            output, _ = process_image(
                model, transform, device, detector, smoother, frame, image_size
            )
            elapsed = max(time.perf_counter() - started, 1e-6)
            fps_history.append(1.0 / elapsed)
            fps = sum(fps_history) / len(fps_history)
            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if args.output:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(args.output),
                        fourcc,
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



