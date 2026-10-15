"""
Детектор лиц на базе YuNet (OpenCV DNN).
"""

import os
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np


DEFAULT_MODEL = "models/face_detection_yunet_2023mar.onnx"


@dataclass
class FaceDetection:
    box: tuple
    score: float
    landmarks: np.ndarray

    @property
    def x1(self): return self.box[0]
    @property
    def y1(self): return self.box[1]
    @property
    def x2(self): return self.box[0] + self.box[2]
    @property
    def y2(self): return self.box[1] + self.box[3]


class FaceDetector:
    """Обёртка над cv2.FaceDetectorYN."""

    def __init__(self,
                 model_path: str = DEFAULT_MODEL,
                 score_threshold: float = 0.9,   # в единицах 0..1
                 nms_threshold: float = 0.3,
                 top_k: int = 100):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Не найдена модель YuNet: {model_path}. "
                f"Скачайте с https://github.com/opencv/opencv_zoo"
            )

        self.score_threshold = score_threshold
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            # Внутренний фильтр работает в 0..1 — доверяем ему.
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Детекция лиц на одном изображении.
        YuNet плохо работает на очень больших входах — ресайзим до 640.
        """
        h, w = image.shape[:2]

        MAX_SIDE = 640
        scale = 1.0
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            small = cv2.resize(image, (new_w, new_h))
        else:
            small = image

        sh, sw = small.shape[:2]
        self.detector.setInputSize((sw, sh))
        ok, faces = self.detector.detect(small)
        if not ok or faces is None:
            return []

        results: List[FaceDetection] = []
        for face in faces:
            # score из OpenCV — не используем для фильтрации,
            # только как метрику для отображения.
            # Приводим к 0..1, если значения выходят за диапазон.
            raw = float(face[4])
            score = raw if 0.0 <= raw <= 1.0 else min(1.0, raw / 500.0)

            x = int(face[0] / scale)
            y = int(face[1] / scale)
            w_box = int(face[2] / scale)
            h_box = int(face[3] / scale)

            # Минимальный размер бокса — отсекаем мусор.
            if w_box < 30 or h_box < 30:
                continue

            landmarks = face[5:15].reshape(5, 2).astype(float) / scale
            results.append(FaceDetection(
                box=(x, y, w_box, h_box),
                score=score,
                landmarks=landmarks,
            ))
        return results

    def detect_batch(self, images: Sequence[np.ndarray]
                     ) -> List[List[FaceDetection]]:
        return [self.detect(img) for img in images]