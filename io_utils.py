"""
Сохранение результатов: кропы, CSV, сводка, сбор входных файлов.
"""

import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Iterable, List

import cv2
import numpy as np

from detector import Detection
from face_detector import FaceDetection
from json_report import ReportBuilder


# ---------- Кропы ----------

def save_crops(image: np.ndarray,
               detections: List[Detection],
               src_name: str,
               crops_dir: str,
               counter: dict) -> int:
    """Сохраняет кропы YOLO-объектов. Имя: {stem}_{class}_{NNN}.jpg"""
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
    """Сохраняет кропы лиц. Имя: {stem}_face_{NNN}.jpg"""
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


# ---------- CSV и сводка ----------

def save_csv(report: ReportBuilder, path: str) -> int:
    """Плоская таблица: одна строка = одна детекция."""
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
    """Сводка по классам: количество, средний score, min/max."""
    stats: dict = defaultdict(list)
    for img in report.images:
        for d in img.detections:
            stats[d["class_name"]].append(d["score"])

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


# ---------- Сбор входных файлов ----------

def collect_images(paths: Iterable[str]) -> List[str]:
    """
    Принимает пути к файлам, папкам и glob-паттернам.
    Возвращает отсортированный список уникальных изображений.
    """
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