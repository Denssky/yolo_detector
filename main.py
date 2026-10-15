# main.py — детектор объектов на изображениях через YOLOv4-tiny и OpenCV DNN.
#
# Возможности:
#   - обработка файлов, папок и glob-паттернов;
#   - фильтр по классам COCO (через --classes или готовые --preset);
#   - батч-инференс с автоматической группировкой по размеру изображения;
#   - сохранение аннотированных картинок, сводного JSON, CSV и кропов;
#   - детекция лиц через YuNet (--faces);
#   - графики matplotlib (--plots);
#   - настройки через JSON-конфиг (--config);
#   - интерактивный просмотр результатов в окне OpenCV.

import os
import sys
import time
import glob
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Sequence
from collections import defaultdict

import cv2
import numpy as np

from json_report import ReportBuilder, ReportMeta, ImageRecord, detection_to_dict
from face_detector import FaceDetector, FaceDetection
from report_plots import save_report_plots
from config_loader import load_config, merge_into_args
from excel_report import save_excel


# ============================================================================
# Константы
# ============================================================================

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
    "all":     (),
}

TEXT_COLOR  = (0, 0, 0)
COUNT_COLOR = (0, 0, 255)
DEFAULT_BOX_COLOR = (0, 255, 0)
FACE_COLOR = (0, 255, 255)
FACE_LANDMARK_COLOR = (0, 0, 255)

CLASS_PALETTE = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (0, 128, 255),
    (128, 128, 0),
    (0, 128, 0),
    (203, 192, 255),
]


def color_for_class(class_id: int) -> tuple:
    return CLASS_PALETTE[class_id % len(CLASS_PALETTE)]


# ============================================================================
# Модель данных
# ============================================================================

@dataclass
class Detection:
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


# ============================================================================
# Детектор YOLO
# ============================================================================

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


# ============================================================================
# Отрисовка
# ============================================================================

def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    for d in detections:
        color = color_for_class(d.class_id)
        cv2.rectangle(image, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"{d.class_name} {d.score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (d.x1, d.y1 - th - base),
                      (d.x1 + tw, d.y1), color, -1)
        cv2.putText(image, label, (d.x1, d.y1 - base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

    cv2.putText(image, f"Detections: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COUNT_COLOR, 2, cv2.LINE_AA)
    return image


def draw_faces(image: np.ndarray, faces: List[FaceDetection]) -> np.ndarray:
    """Рисует боксы лиц и 5 ключевых точек."""
    for f in faces:
        cv2.rectangle(image, (f.x1, f.y1), (f.x2, f.y2), FACE_COLOR, 2)

        label = f"face {f.score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (f.x1, f.y1 - th - base),
                      (f.x1 + tw, f.y1), FACE_COLOR, -1)
        cv2.putText(image, label, (f.x1, f.y1 - base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        for (lx, ly) in f.landmarks:
            cv2.circle(image, (int(lx), int(ly)), 2, FACE_LANDMARK_COLOR, -1)
    return image


# ============================================================================
# Сохранение результатов
# ============================================================================

def save_crops(image: np.ndarray,
               detections: List[Detection],
               src_name: str,
               crops_dir: str,
               counter: dict) -> int:
    if not detections:
        return 0

    os.makedirs(crops_dir, exist_ok=True)
    stem = os.path.splitext(src_name)[0]
    h_img, w_img = image.shape[:2]
    saved = 0

    for d in detections:
        x1 = max(0, d.x1); y1 = max(0, d.y1)
        x2 = min(w_img, d.x2); y2 = min(h_img, d.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        class_slug = d.class_name.replace(" ", "_")
        key = (stem, class_slug)
        counter[key] = counter.get(key, 0) + 1
        num = counter[key]

        fname = f"{stem}_{class_slug}_{num:03d}.jpg"
        cv2.imwrite(os.path.join(crops_dir, fname), crop)
        saved += 1

    return saved


def save_face_crops(image: np.ndarray,
                    faces: List[FaceDetection],
                    src_name: str,
                    crops_dir: str,
                    counter: dict) -> int:
    if not faces:
        return 0
    os.makedirs(crops_dir, exist_ok=True)
    stem = os.path.splitext(src_name)[0]
    h_img, w_img = image.shape[:2]
    saved = 0
    for f in faces:
        x1 = max(0, f.x1); y1 = max(0, f.y1)
        x2 = min(w_img, f.x2); y2 = min(h_img, f.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        key = (stem, "face")
        counter[key] = counter.get(key, 0) + 1
        num = counter[key]
        fname = f"{stem}_face_{num:03d}.jpg"
        cv2.imwrite(os.path.join(crops_dir, fname), crop)
        saved += 1
    return saved


def save_csv(report: ReportBuilder, path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "class_name", "class_id",
                         "score", "x", "y", "w", "h"])
        for img in report.images:
            for d in img.detections:
                box = d["box"]
                writer.writerow([
                    img.path,
                    d["class_name"],
                    d["class_id"],
                    d["score"],
                    box["x"], box["y"], box["w"], box["h"],
                ])
                rows += 1
    return rows


def print_class_summary(report: ReportBuilder) -> None:
    stats: dict = {}
    for img in report.images:
        for d in img.detections:
            stats.setdefault(d["class_name"], []).append(d["score"])

    print("\n--- Сводка по классам ---")
    if not stats:
        print("  (ничего не найдено)")
        return

    items = sorted(stats.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    name_w = max(len(name) for name, _ in items)
    total = 0
    for name, scores in items:
        n = len(scores)
        total += n
        avg = sum(scores) / n
        lo = min(scores); hi = max(scores)
        print(f"  {name:<{name_w}} : {n:>3}  "
              f"avg={avg:.2f}  ({lo:.2f} .. {hi:.2f})")
    print(f"Всего: {total} детекций, {len(items)} класс(ов)")


# ============================================================================
# Сбор файлов и CLI
# ============================================================================

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


def build_parser():
    """Создаёт argparse-парсер со всеми аргументами."""
    p = argparse.ArgumentParser(description="YOLOv4-tiny object detector (images)")
    p.add_argument("inputs", nargs="*", default=["images/trucks.jpg"],
                   help="файлы, папки или glob-паттерны "
                        "(по умолчанию: images/trucks.jpg)")
    p.add_argument("--config", default=None,
                   help="путь к JSON-файлу с настройками")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--json", default="output/detections.json")
    p.add_argument("--csv", default="output/detections.csv",
                   help="путь к CSV (пустая строка — отключить)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--nms", type=float, default=DEFAULT_NMS)
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    p.add_argument("--preset", choices=list(CLASS_PRESETS.keys()))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--crops-dir", default=None)
    p.add_argument("--faces", action="store_true")
    p.add_argument("--face-model",
                   default="models/face_detection_yunet_2023mar.onnx")
    p.add_argument("--face-conf", type=float, default=0.7)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--plots-path", default="output/report.png")
    p.add_argument("--xlsx", default="output/detections.xlsx",
                   help="путь к Excel-отчёту (пустая строка — отключить)")
    return p


def parse_args():
    """
    Разбор аргументов. Если указан --config, значения из JSON применяются
    как дефолты, но CLI-флаги остаются приоритетнее.
    """
    # 1. Предварительный парсинг: только --config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()

    config = {}
    if known.config:
        try:
            config = load_config(known.config)
            print(f"Конфиг: {known.config}  ({len(config)} ключей)")
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] {e}", file=sys.stderr)
            print("[warn] --config игнорируется", file=sys.stderr)

    # 2. Полный парсинг
    parser = build_parser()
    defaults = parser.parse_args([])
    args = parser.parse_args()

    # 3. Мержим конфиг: CLI побеждает, конфиг заполняет «дырки»
    if config:
        args, applied = merge_into_args(args, config, defaults)
        if applied:
            print(f"Из конфига применено: {', '.join(applied)}")

    return args


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ============================================================================
# Интерактивный просмотр
# ============================================================================

def browse_results(results: List[Tuple[str, np.ndarray]],
                   window_name: str = "YOLOv4-tiny") -> None:
    n = len(results)
    idx = 0
    print(f"\nПросмотр {n} изображений. "
          f"A/[ — назад, D/]/Space — вперёд, Q — выход.")

    while True:
        name, img = results[idx]
        display = img.copy()
        hud = f"[{idx + 1}/{n}]  {name}"
        y = display.shape[0] - 15
        cv2.putText(display, hud, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, hud, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(0)
        if key == -1:
            continue
        low = key & 0xFF

        if low in (ord("q"), ord("Q"), 27):
            break
        if low in (ord("d"), ord("D"), 32, ord("]")):
            idx = (idx + 1) % n
        elif low in (ord("a"), ord("A"), ord("[")):
            idx = (idx - 1) % n

    cv2.destroyAllWindows()


# ============================================================================
# Точка входа
# ============================================================================

def main():
    args = parse_args()

    # Если inputs пуст — берём дефолт
    if not args.inputs:
        args.inputs = ["images/trucks.jpg"]

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

    crops_dir = args.crops_dir or os.path.join(args.output_dir, "crops")
    crops_counter: dict = {}
    total_crops = 0
    if args.save_crops:
        os.makedirs(crops_dir, exist_ok=True)
        print(f"Кропы: {crops_dir}")

    face_detector = None
    if args.faces:
        try:
            face_detector = FaceDetector(
                model_path=args.face_model,
                score_threshold=args.face_conf,
            )
            print(f"Face-детектор: {args.face_model}")
        except FileNotFoundError as e:
            print(f"[warn] {e}", file=sys.stderr)
            print("[warn] --faces игнорируется", file=sys.stderr)

    report = ReportBuilder(meta=ReportMeta(
        model=os.path.basename(MODEL_DIR),
        input_size=INPUT_SIZE,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        classes_of_interest=tuple(classes),
        batch_size=args.batch_size,
    ))

    total_time = 0.0
    saved_results: List[Tuple[str, np.ndarray]] = []

    by_size: dict = defaultdict(list)
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"[warn] не удалось прочитать: {f}", file=sys.stderr)
            continue
        by_size[img.shape[:2]].append((f, img))

    if not by_size:
        print("Не удалось прочитать ни одного изображения.", file=sys.stderr)
        sys.exit(1)

    sizes_info = ", ".join(f"{h}x{w}: {len(v)}"
                           for (h, w), v in by_size.items())
    print(f"Групп по размеру: {len(by_size)} ({sizes_info})")

    batch_idx = 0
    for (grp_h, grp_w), items in by_size.items():
        for batch_files in chunked(items, args.batch_size):
            batch_idx += 1
            valid_files = [f for f, _ in batch_files]
            images = [img for _, img in batch_files]

            t0 = time.perf_counter()
            batch_dets = detector.detect_batch(images)
            dt_batch = (time.perf_counter() - t0) * 1000
            total_time += dt_batch
            per_img_ms = dt_batch / len(images)

            print(f"[batch {batch_idx}] group {grp_h}x{grp_w}  "
                  f"{len(images)} img  "
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

                faces: List[FaceDetection] = []
                if face_detector is not None:
                    faces = face_detector.detect(img)
                    if args.verbose and faces:
                        for fd in faces:
                            print(f"      {'face':<12} "
                                  f"score={fd.score:.3f} box={fd.box}")
                    result = draw_faces(result, faces)

                    count_text = f"Faces: {len(faces)}"
                    (tw2, _), _ = cv2.getTextSize(
                        count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(result, count_text,
                                (result.shape[1] - tw2 - 10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                FACE_COLOR, 2, cv2.LINE_AA)

                out_path = os.path.join(args.output_dir, name)
                cv2.imwrite(out_path, result)
                saved_results.append((name, result))

                if args.save_crops:
                    total_crops += save_crops(img, dets, name,
                                              crops_dir, crops_counter)
                    if faces:
                        total_crops += save_face_crops(
                            img, faces, name, crops_dir, crops_counter)

                all_dets = [detection_to_dict(d) for d in dets]
                for fd in faces:
                    all_dets.append({
                        "class_id": -1,
                        "class_name": "face",
                        "score": round(float(fd.score), 4),
                        "box": {"x": int(fd.x1), "y": int(fd.y1),
                                "w": int(fd.box[2]), "h": int(fd.box[3])},
                        "xyxy": [int(fd.x1), int(fd.y1),
                                 int(fd.x2), int(fd.y2)],
                    })

                report.add_image(ImageRecord(
                    path=f,
                    output=out_path,
                    width=w,
                    height=h,
                    inference_ms=per_img_ms,
                    detections=all_dets,
                ))

    if args.save_crops:
        print(f"Кропов сохранено: {total_crops}")

    report.meta.total_time_ms = total_time
    saved = report.save(args.json)

    if args.csv:
        if args.xlsx:
            xlsx_path = save_excel(report, args.xlsx)
        if xlsx_path:
            print(f"XLSX: {xlsx_path}")
        n_rows = save_csv(report, args.csv)
        print(f"CSV: {args.csv}  ({n_rows} строк)")

    print_class_summary(report)

    if args.plots:
        plot_path = save_report_plots(report, args.plots_path)
        if plot_path:
            print(f"Графики: {plot_path}")

    total_dets = report.meta.total_detections
    print(f"\nИтого: {report.meta.total_images} файлов, "
          f"{total_dets} детекций, "
          f"{total_time:.1f} ms  "
          f"({total_time / max(report.meta.total_images, 1):.1f} ms/img)")
    print(f"JSON: {saved}")

    if args.show and saved_results:
        try:
            browse_results(saved_results)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()