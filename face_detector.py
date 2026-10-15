"""
Детектор лиц на базе YuNet (OpenCV DNN).
Лёгкая ONNX-модель, работает на CPU быстро (~5-15 мс на кадр 640x480).
"""

import os
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np


# Модель по умолчанию лежит в models/
DEFAULT_MODEL = "models/face_detection_yunet_2023mar.onnx"


@dataclass
class FaceDetection:
    """
    Одно лицо: бокс, уверенность, 5 ключевых точек.
    box — (x, y, w, h) в пикселях.
    landmarks — массив (5, 2): [правый глаз, левый глаз, нос,
    правый уголок рта, левый уголок рта].
    """
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
                 score_threshold: float = 0.7,
                 nms_threshold: float = 0.3,
                 top_k: int = 5000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Не найдена модель YuNet: {model_path}. "
                f"Скачайте с https://github.com/opencv/opencv_zoo"
            )

        self.score_threshold = score_threshold
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),   # будет переопределяться в detect()
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Детекция лиц на одном изображении.

        YuNet плохо работает на очень больших входах, поэтому ресайзим
        изображение до 640 пикселей по длинной стороне, а координаты
        детекций масштабируем обратно к оригинальному размеру.
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
            # score — приводим к float, фильтруем вручную:
            # OpenCV может игнорировать score_threshold в create().
            score = float(face[4])
            if score < self.score_threshold:
                continue

            # Координаты конвертируем в обычные int, чтобы json.dump
            # мог их сериализовать (np.int64 → TypeError).
            # Делим на scale — возвращаем к оригинальному размеру.
            x = int(face[0] / scale)
            y = int(face[1] / scale)
            w_box = int(face[2] / scale)
            h_box = int(face[3] / scale)

            # Отсекаем мусорные мелкие боксы.
            if w_box < 20 or h_box < 20:
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
        """
        YuNet не поддерживает настоящий батч через OpenCV, поэтому
        просто вызываем detect() по очереди. На CPU это ~5-15 мс на кадр.
        """
        return [self.detect(img) for img in images]