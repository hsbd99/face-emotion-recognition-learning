"""OpenCV Zoo pretrained facial-expression model wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import EMOTIONS, PROJECT_ROOT
from face_detector import FaceDetection

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "facial_expression_recognition_mobilefacenet_2022july.onnx"
MODEL_CLASSES = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
MODEL_TO_PROJECT_ORDER = np.array([0, 1, 2, 3, 6, 4, 5])
STANDARD_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class OpenCVFacialExpressionRecognizer:
    """Run the OpenCV Zoo MobileFaceNet FER model on CPU or another OpenCV backend."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Pretrained FER model not found: {self.model_path}")
        self.model = cv2.dnn.readNet(str(self.model_path))
        self.model.setPreferableBackend(backend_id)
        self.model.setPreferableTarget(target_id)

    def align_face(self, frame: np.ndarray, detection: FaceDetection | None) -> np.ndarray:
        if detection is not None and detection.keypoints:
            points = np.asarray(
                [
                    detection.keypoints["right_eye"],
                    detection.keypoints["left_eye"],
                    detection.keypoints["nose"],
                    detection.keypoints["right_mouth"],
                    detection.keypoints["left_mouth"],
                ],
                dtype=np.float32,
            )
            matrix, _ = cv2.estimateAffinePartial2D(points, STANDARD_LANDMARKS, method=cv2.LMEDS)
            if matrix is not None:
                return cv2.warpAffine(frame, matrix, (112, 112), flags=cv2.INTER_LINEAR)

        if detection is not None:
            height, width = frame.shape[:2]
            pad_x = int(detection.width * 0.1)
            pad_y = int(detection.height * 0.1)
            x1 = max(0, detection.x1 - pad_x)
            y1 = max(0, detection.y1 - pad_y)
            x2 = min(width, detection.x2 + pad_x)
            y2 = min(height, detection.y2 + pad_y)
            face = frame[y1:y2, x1:x2]
        else:
            face = frame
        if face.size == 0:
            face = frame
        return cv2.resize(face, (112, 112))

    def _blob(self, face: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        return cv2.dnn.blobFromImage(rgb)

    def infer_probs(self, frame: np.ndarray, detection: FaceDetection | None = None) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Cannot run expression recognition on an empty frame")
        face = self.align_face(frame, detection)
        self.model.setInput(self._blob(face), "data")
        logits = self.model.forward(["label"])[0]
        exp_logits = np.exp(logits - logits.max())
        raw_probabilities = (exp_logits / exp_logits.sum()).reshape(-1)
        project_probabilities = np.zeros_like(raw_probabilities)
        project_probabilities[MODEL_TO_PROJECT_ORDER] = raw_probabilities
        return project_probabilities

    def infer(self, frame: np.ndarray, detection: FaceDetection | None = None) -> tuple[str, float]:
        probabilities = self.infer_probs(frame, detection)
        index = int(np.argmax(probabilities))
        return EMOTIONS[index], float(probabilities[index])


__all__ = [
    "DEFAULT_MODEL_PATH",
    "MODEL_CLASSES",
    "OpenCVFacialExpressionRecognizer",
]


