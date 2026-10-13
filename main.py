import os
import sys
import time
import glob
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Sequence

import cv2
import numpy as np

from json_report import ReportBuilder, ReportMeta, ImageRecord, detection_to_dict


# ---------- Константы ----------
MODEL_DIR = "yolo_tiny"
WEIGHTS   = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
CONFIG    = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
NAMES     = os.path.join(MODEL_DIR, "coco.names")

INPUT_SIZE      = (416, 416)
DEFAULT_CONF    = 0.4
DEFAULT_NMS     = 0.4
DEFAULT_CLASSES = ("truck",)
CLASS_PRESETS = {
    "truck":   ("truck",),
    "traffic": ("person", "car", "bus", "truck", "motorcycle", "bicycle",
                "traffic light", "stop sign"),
    "office":  ("person", "chair", "laptop", "keyboard", "mouse",
                "cell phone", "book", "clock", "cup"),
    "food":    ("banana", "apple", "sandwich", "orange", "pizza",
                "donut", "cake", "bottle", "cup", "bowl", "fork", "knife"),
    "all":     (),   # пустой = все 80 классов
}

BOX_COLOR   = (0, 255, 0)
TEXT_COLOR  = (0, 0, 0)
COUNT_COLOR = (0, 0, 255)

@dataclass
class Detection:
    box: Tuple[int, int, int, int]   # x, y, w, h
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

        self.net = cv2.dnn.readNet(weights, config)
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


def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    for d in detections:
        cv2.rectangle(image, (d.x1, d.y1), (d.x2, d.y2), BOX_COLOR, 2)
        label = f"{d.class_name} {d.score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (d.x1, d.y1 - th - base),
                      (d.x1 + tw, d.y1), BOX_COLOR, -1)
        cv2.putText(image, label, (d.x1, d.y1 - base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

    cv2.putText(image, f"Trucks found: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COUNT_COLOR, 2, cv2.LINE_AA)
    return image


def collect_images(paths: Iterable[str]) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    found: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in exts:
                found.extend(glob.glob(os.path.join(p, f"*{ext}")))
                found.extend(glob.glob(os.path.join(p, f"*{ext.upper()}")))
        elif any(ch in p for ch in "*?["):
            found.extend(glob.glob(p, recursive=True))
        elif os.path.isfile(p):
            found.append(p)
        else:
            print(f"[warn] не найдено: {p}", file=sys.stderr)

    seen = set()
    unique = []
    for f in sorted(found):
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv4-tiny truck detector (images)")
    p.add_argument("inputs", nargs="*", default=["images/trucks.jpg"],
                   help="файлы, папки или glob-паттерны "
                        "(по умолчанию: images/trucks.jpg)")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--json", default="output/detections.json",
                   help="путь к сводному JSON")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--nms", type=float, default=DEFAULT_NMS)
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    p.add_argument("--preset", choices=list(CLASS_PRESETS.keys()),
                   help="готовый набор классов (перебивает --classes)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--show", action="store_true",
                   help="открыть окно с результатом (по умолчанию — нет)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    args = parse_args()

    if args.preset:
        classes = CLASS_PRESETS[args.preset]
    else:
        classes = tuple(args.classes)

    detector = YoloDetector(
        WEIGHTS, CONFIG, NAMES,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        classes_of_interest=classes,
        use_cuda=args.cuda,
    )

    files = collect_images(args.inputs)
    if not files:
        print("Не найдено ни одного изображения.", file=sys.stderr)
        sys.exit(1)

    print(f"Классов: {len(detector.classes)}  |  файлов: {len(files)}")
    print(f"conf={args.conf}  nms={args.nms}  batch_size={args.batch_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    report = ReportBuilder(meta=ReportMeta(
        model=os.path.basename(MODEL_DIR),
        input_size=INPUT_SIZE,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        classes_of_interest=tuple(args.classes),
        batch_size=args.batch_size,
    ))
    total_time = 0.0
    last_result = None

    for batch_idx, batch_files in enumerate(chunked(files, args.batch_size), 1):
        images, valid_files = [], []
        for f in batch_files:
            img = cv2.imread(f)
            if img is None:
                print(f"[warn] не удалось прочитать: {f}", file=sys.stderr)
                continue
            images.append(img)
            valid_files.append(f)

        if not images:
            continue

        t0 = time.perf_counter()
        batch_dets = detector.detect_batch(images)
        dt_batch = (time.perf_counter() - t0) * 1000
        total_time += dt_batch
        per_img_ms = dt_batch / len(images)

        print(f"[batch {batch_idx}] {len(images)} img  "
              f"{dt_batch:.1f} ms total  ({per_img_ms:.1f} ms/img)")

        for f, img, dets in zip(valid_files, images, batch_dets):
            h, w = img.shape[:2]
            name = os.path.basename(f)
            print(f"  {name:<30} найдено: {len(dets)}")

            if args.verbose:
                for d in sorted(dets, key=lambda x: -x.score):
                    print(f"      {d.class_name:<12} "
                          f"score={d.score:.3f} box={d.box}")

            result = draw_detections(img, dets)
            out_path = os.path.join(args.output_dir, name)
            cv2.imwrite(out_path, result)
            last_result = result

            report.add_image(ImageRecord(
                path=f,
                output=out_path,
                width=w,
                height=h,
                inference_ms=per_img_ms,
                detections=[detection_to_dict(d) for d in dets],
            ))

    report.meta.total_time_ms = total_time
    saved = report.save(args.json)

    total_dets = report.meta.total_detections
    print(f"\nИтого: {report.meta.total_images} файлов, "
          f"{total_dets} детекций, "
          f"{total_time:.1f} ms  "
          f"({total_time / max(report.meta.total_images, 1):.1f} ms/img)")
    print(f"JSON: {saved}")

    if not args.no_show and last_result is not None:
        cv2.imshow("YOLOv4-tiny", last_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()