"""Face detection abstraction with YuNet, optional MTCNN and Haar fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import PROJECT_ROOT

YUNET_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


@dataclass
class FaceDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    keypoints: dict[str, tuple[int, int]] | None = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class FaceDetector:
    """Prefer OpenCV YuNet, then MTCNN, then the bundled Haar cascade."""

    def __init__(
        self,
        backend: str = "auto",
        min_face_size: int = 40,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        yunet_model: str | Path = YUNET_MODEL_PATH,
    ) -> None:
        if backend not in {"auto", "yunet", "mtcnn", "haar"}:
            raise ValueError("backend must be one of: auto, yunet, mtcnn, haar")
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.mtcnn: Any | None = None
        self.yunet: Any | None = None
        self.backend = backend

        if backend in {"auto", "yunet"}:
            model_path = Path(yunet_model)
            if model_path.is_file():
                try:
                    self.yunet = cv2.FaceDetectorYN.create(
                        str(model_path),
                        "",
                        (320, 320),
                        score_threshold=0.75,
                        nms_threshold=0.3,
                        top_k=5000,
                    )
                    self.backend = "yunet"
                except Exception:
                    if backend == "yunet":
                        raise
            elif backend == "yunet":
                raise FileNotFoundError(f"YuNet model not found: {model_path}")

        if self.backend == "auto":
            try:
                from mtcnn import MTCNN

                self.mtcnn = MTCNN(min_face_size=min_face_size)
                self.backend = "mtcnn"
            except Exception:
                self.backend = "haar"
        elif backend == "mtcnn":
            from mtcnn import MTCNN

            self.mtcnn = MTCNN(min_face_size=min_face_size)
            self.backend = "mtcnn"

        if self.backend == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                raise RuntimeError(f"Could not load OpenCV face cascade: {cascade_path}")

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        height, width = frame.shape[:2]
        longest = max(height, width)
        scale = min(4.0, 320.0 / longest) if longest < 320 else 1.0
        working = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else frame

        if self.backend == "yunet" and self.yunet is not None:
            detections = self._detect_yunet(working)
        elif self.backend == "mtcnn" and self.mtcnn is not None:
            detections = self._detect_mtcnn(working)
        else:
            detections = self._detect_haar(working)

        if scale == 1.0:
            return detections
        return [self._scale_detection(detection, 1.0 / scale) for detection in detections]

    @staticmethod
    def _scale_detection(detection: FaceDetection, scale: float) -> FaceDetection:
        keypoints = None
        if detection.keypoints:
            keypoints = {
                name: (int(point[0] * scale), int(point[1] * scale))
                for name, point in detection.keypoints.items()
            }
        return FaceDetection(
            int(detection.x1 * scale),
            int(detection.y1 * scale),
            int(detection.x2 * scale),
            int(detection.y2 * scale),
            detection.confidence,
            keypoints,
        )

    def _detect_yunet(self, frame: np.ndarray) -> list[FaceDetection]:
        height, width = frame.shape[:2]
        self.yunet.setInputSize((width, height))
        _, faces = self.yunet.detect(frame)
        if faces is None:
            return []

        detections: list[FaceDetection] = []
        for face in faces:
            x, y, box_width, box_height = face[:4]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(width, int(x + box_width))
            y2 = min(height, int(y + box_height))
            if x2 <= x1 or y2 <= y1 or min(x2 - x1, y2 - y1) < self.min_face_size:
                continue
            landmarks = face[4:14].reshape(5, 2).astype(int)
            keypoints = {
                "right_eye": tuple(landmarks[0]),
                "left_eye": tuple(landmarks[1]),
                "nose": tuple(landmarks[2]),
                "right_mouth": tuple(landmarks[3]),
                "left_mouth": tuple(landmarks[4]),
            }
            detections.append(FaceDetection(x1, y1, x2, y2, float(face[-1]), keypoints))
        return detections

    def _detect_mtcnn(self, frame: np.ndarray) -> list[FaceDetection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mtcnn.detect_faces(rgb)
        detections: list[FaceDetection] = []
        for result in results:
            x, y, width, height = result["box"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(frame.shape[1], int(x + width))
            y2 = min(frame.shape[0], int(y + height))
            if x2 <= x1 or y2 <= y1:
                continue
            keypoints = {
                name: (int(point[0]), int(point[1]))
                for name, point in result.get("keypoints", {}).items()
            }
            detections.append(
                FaceDetection(x1, y1, x2, y2, float(result.get("confidence", 1.0)), keypoints or None)
            )
        return detections

    def _detect_haar(self, frame: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )
        return [
            FaceDetection(int(x), int(y), int(x + width), int(y + height))
            for x, y, width, height in faces
        ]

    def crop(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        output_size: int = 96,
        margin: float = 0.08,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        pad_x = int(detection.width * margin)
        pad_y = int(detection.height * margin)
        x1 = max(0, detection.x1 - pad_x)
        y1 = max(0, detection.y1 - pad_y)
        x2 = min(width, detection.x2 + pad_x)
        y2 = min(height, detection.y2 + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        return cv2.resize(crop, (output_size, output_size))


__all__ = ["FaceDetection", "FaceDetector", "YUNET_MODEL_PATH"]

