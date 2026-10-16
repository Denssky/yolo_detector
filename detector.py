"""
Модель данных и детектор YOLO.
Не зависит от других модулей проекта.
"""

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


# ---------- Константы модели ----------

MODEL_DIR  = "yolo_tiny"
WEIGHTS    = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
CONFIG     = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
NAMES      = os.path.join(MODEL_DIR, "coco.names")
ONNX_MODEL = os.path.join(MODEL_DIR, "yolov4-tiny.onnx")

INPUT_SIZE      = (416, 416)
DEFAULT_CONF    = 0.4
DEFAULT_NMS     = 0.4
DEFAULT_CLASSES = ("truck",)

# Готовые наборы классов для --preset.
# Пустой кортеж = все 80 классов COCO.
CLASS_PRESETS = {
    "truck":   ("truck",),
    "traffic": ("person", "car", "bus", "truck", "motorcycle", "bicycle",
                "traffic light", "stop sign"),
    "office":  ("person", "chair", "laptop", "keyboard", "mouse",
                "cell phone", "book", "clock", "cup"),
    "food":    ("banana", "apple", "sandwich", "orange", "pizza",
                "donut", "cake", "bottle", "cup", "bowl", "fork", "knife"),
    "all":     (),
}


# ---------- Модель данных ----------

@dataclass
class Detection:
    """
    Одна YOLO-детекция.
    box — (x, y, w, h), (x, y) — левый верхний угол.
    """
    box: Tuple[int, int, int, int]
    class_id: int
    score: float
    class_name: str = ""

    @property
    def x1(self): return self.box[0]
    @property
    def y1(self): return self.box[1]
    @property
    def x2(self): return self.box[0] + self.box[2]
    @property
    def y2(self): return self.box[1] + self.box[3]


# ---------- Детектор ----------

class YoloDetector:
    """Обёртка над cv2.dnn для YOLO."""

    def __init__(self, weights: str, config: str, names: str,
                 input_size=INPUT_SIZE,
                 conf_threshold=DEFAULT_CONF,
                 nms_threshold=DEFAULT_NMS,
                 classes_of_interest=DEFAULT_CLASSES,
                 use_cuda: bool = False):
        for p in (weights, config, names):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Не найден файл: {p}")

        self.net = cv2.dnn.readNetFromONNX("yolo_tiny/yolov4-tiny.onnx")
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes_of_interest = set(classes_of_interest)

        if use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        self.out_layers = [layer_names[i - 1]
                           for i in self.net.getUnconnectedOutLayers()]

        with open(names, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f if line.strip()]

    def _filter_class_ids(self) -> set:
        if not self.classes_of_interest:
            return set(range(len(self.classes)))
        return {i for i, name in enumerate(self.classes)
                if name in self.classes_of_interest}

    def _postprocess(self, outs,
                     orig_sizes: Sequence[Tuple[int, int]]
                     ) -> List[List[Detection]]:
        batch_size = len(orig_sizes)
        per_image_boxes: List[List[List[int]]] = [[] for _ in range(batch_size)]
        per_image_scores: List[List[float]] = [[] for _ in range(batch_size)]
        per_image_class_ids: List[List[int]] = [[] for _ in range(batch_size)]

        for out in outs:
            if out.ndim != 2:
                out = out.reshape(out.shape[0], -1)
            if out.shape[1] != 85 and out.shape[0] == 85:
                out = out.T
            if out.shape[1] != 85:
                continue

            rows = out.shape[0]
            if rows % batch_size != 0:
                continue
            anchors = rows // batch_size

            for b in range(batch_size):
                h, w = orig_sizes[b]
                chunk = out[b * anchors:(b + 1) * anchors]

                class_scores = chunk[:, 5:]
                class_ids = np.argmax(class_scores, axis=1)
                scores = class_scores[np.arange(len(class_ids)), class_ids]
                mask = scores >= self.conf_threshold
                if not mask.any():
                    continue

                det = chunk[mask]
                cls = class_ids[mask]
                sc = scores[mask]

                cx = det[:, 0] * w
                cy = det[:, 1] * h
                bw = det[:, 2] * w
                bh = det[:, 3] * h
                x = np.maximum(0, cx - bw / 2).astype(int)
                y = np.maximum(0, cy - bh / 2).astype(int)
                bw = bw.astype(int)
                bh = bh.astype(int)

                per_image_boxes[b].extend(
                    np.stack([x, y, bw, bh], axis=1).tolist())
                per_image_scores[b].extend(sc.tolist())
                per_image_class_ids[b].extend(cls.tolist())

        wanted = self._filter_class_ids()
        results: List[List[Detection]] = []
        for b in range(batch_size):
            boxes = per_image_boxes[b]
            scores = per_image_scores[b]
            class_ids = per_image_class_ids[b]
            if not boxes:
                results.append([])
                continue

            idxs = cv2.dnn.NMSBoxes(boxes, scores,
                                    self.conf_threshold, self.nms_threshold)
            if len(idxs) == 0:
                results.append([])
                continue

            dets = []
            for i in np.array(idxs).flatten():
                i = int(i)
                cid = class_ids[i]
                if cid not in wanted:
                    continue
                dets.append(Detection(
                    box=tuple(boxes[i]),
                    class_id=cid,
                    score=scores[i],
                    class_name=(self.classes[cid]
                                if cid < len(self.classes) else "unknown"),
                ))
            results.append(dets)
        return results

    def detect(self, image: np.ndarray) -> List[Detection]:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]
                     ) -> List[List[Detection]]:
        if not images:
            return []
        orig_sizes = [img.shape[:2] for img in images]
        blob = cv2.dnn.blobFromImages(
            list(images), 1 / 255.0, self.input_size, (0, 0, 0),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)
        return self._postprocess(outs, orig_sizes)
