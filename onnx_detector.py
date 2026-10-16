"""
Детектор на базе ONNX Runtime (альтернатива OpenCV DNN).
Формат выходов: boxes (N, 2535, 1, 4) xyxy в [0..1],
confs (N, 2535, 80) с сигмоидой.
"""

import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from detector import Detection


class OnnxDetector:
    """Обёртка над onnxruntime.InferenceSession для YOLO."""

    def __init__(self,
                 model_path: str,
                 names_path: str,
                 input_size: Tuple[int, int] = (416, 416),
                 conf_threshold: float = 0.4,
                 nms_threshold: float = 0.4,
                 classes_of_interest: Sequence[str] = ("truck",),
                 use_cuda: bool = False):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Не найден файл: {model_path}")

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_cuda else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes_of_interest = set(classes_of_interest)

        with open(names_path, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f if line.strip()]

    def _filter_class_ids(self) -> set:
        if not self.classes_of_interest:
            return set(range(len(self.classes)))
        return {i for i, name in enumerate(self.classes)
                if name in self.classes_of_interest}

    def _preprocess(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """BGR, resize, HWC -> CHW, /255, batch-ось."""
        n = len(images)
        w, h = self.input_size
        batch = np.zeros((n, 3, h, w), dtype=np.float32)
        for i, img in enumerate(images):
            resized = cv2.resize(img, (w, h))
            batch[i] = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return batch

    def detect_batch(self, images: Sequence[np.ndarray]
                     ) -> List[List[Detection]]:
        if not images:
            return []

        orig_sizes = [img.shape[:2] for img in images]
        blob = self._preprocess(images)

        boxes, confs = self.session.run(None, {self.input_name: blob})

        batch_size = len(images)
        wanted = self._filter_class_ids()
        results: List[List[Detection]] = []

        for b in range(batch_size):
            h, w = orig_sizes[b]
            b_boxes = boxes[b].reshape(-1, 4)
            b_confs = confs[b]

            class_ids = np.argmax(b_confs, axis=1)
            scores = b_confs[np.arange(len(class_ids)), class_ids]

            mask = scores >= self.conf_threshold
            if not mask.any():
                results.append([])
                continue

            sel_boxes = b_boxes[mask]
            sel_scores = scores[mask]
            sel_class_ids = class_ids[mask]

            # xyxy (норм. 0..1) -> xywh (пиксели)
            x1 = sel_boxes[:, 0] * w
            y1 = sel_boxes[:, 1] * h
            x2 = sel_boxes[:, 2] * w
            y2 = sel_boxes[:, 3] * h
            x = np.maximum(0, x1)
            y = np.maximum(0, y1)
            bw = np.maximum(0, x2 - x1)
            bh = np.maximum(0, y2 - y1)

            nms_boxes = [
                (int(x[i]), int(y[i]), int(bw[i]), int(bh[i]))
                for i in range(len(x))
            ]
            nms_scores = [float(s) for s in sel_scores]

            idxs = cv2.dnn.NMSBoxes(nms_boxes, nms_scores,
                                    self.conf_threshold, self.nms_threshold)
            if len(idxs) == 0:
                results.append([])
                continue

            dets = []
            for i in np.array(idxs).flatten():
                i = int(i)
                cid = int(sel_class_ids[i])
                if cid not in wanted:
                    continue
                dets.append(Detection(
                    box=tuple(nms_boxes[i]),
                    class_id=cid,
                    score=float(nms_scores[i]),
                    class_name=(self.classes[cid]
                                if cid < len(self.classes) else "unknown"),
                ))
            results.append(dets)

        return results

    def detect(self, image: np.ndarray) -> List[Detection]:
        return self.detect_batch([image])[0]

    def detect_tiled(self,
                     image: np.ndarray,
                     tile_size: int = 640,
                     overlap: int = 128) -> List[Detection]:
        """Детекция с тайлингом (см. YoloDetector.detect_tiled)."""
        from tiling import compute_tiles, global_nms

        h, w = image.shape[:2]
        if w <= tile_size and h <= tile_size:
            return self.detect(image)

        tiles = compute_tiles(w, h, tile_size, overlap)
        all_dets: List[Detection] = []

        for (x1, y1, x2, y2) in tiles:
            crop = image[y1:y2, x1:x2]
            tile_dets = self.detect(crop)
            for d in tile_dets:
                all_dets.append(Detection(
                    box=(d.x1 + x1, d.y1 + y1, d.box[2], d.box[3]),
                    class_id=d.class_id,
                    score=d.score,
                    class_name=d.class_name,
                ))

        return global_nms(all_dets, self.nms_threshold)