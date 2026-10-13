"""
Сбор и сохранение сводного JSON-отчёта по детекциям.

Формат:
{
  "meta": { ... },
  "images": [
    {
      "path": ..., "output": ...,
      "width": ..., "height": ..., "inference_ms": ...,
      "detections": [ { class_id, class_name, score, box, xyxy }, ... ]
    }, ...
  ]
}
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ---------- DTO ----------
@dataclass
class ImageRecord:
    """Результат по одному изображению."""
    path: str
    output: Optional[str] = None
    width: int = 0
    height: int = 0
    inference_ms: float = 0.0
    detections: List[dict] = field(default_factory=list)


@dataclass
class ReportMeta:
    """Метаданные запуска."""
    model: str = ""
    input_size: tuple = (416, 416)
    conf_threshold: float = 0.0
    nms_threshold: float = 0.0
    classes_of_interest: tuple = ()
    batch_size: int = 1
    timestamp: str = ""
    total_images: int = 0
    total_detections: int = 0
    total_time_ms: float = 0.0


# ---------- Сборщик ----------
class ReportBuilder:
    """
    Аккумулятор результатов. Используется в main():

        rb = ReportBuilder(meta=ReportMeta(...))
        rb.add_image(ImageRecord(...))
        rb.save("output/detections.json")
    """

    def __init__(self, meta: Optional[ReportMeta] = None):
        self.meta = meta or ReportMeta()
        self.images: List[ImageRecord] = []

    def add_image(self, record: ImageRecord) -> None:
        self.images.append(record)
        self.meta.total_images = len(self.images)
        self.meta.total_detections = sum(len(r.detections) for r in self.images)

    def add_detection_dicts(self, path: str, dets: List[dict], **kwargs) -> None:
        """Удобный шорткат, если detections уже в виде dict."""
        self.add_image(ImageRecord(path=path, detections=dets, **kwargs))

    def _to_dict(self) -> dict:
        meta = {
            "model": self.meta.model,
            "input_size": list(self.meta.input_size),
            "conf_threshold": self.meta.conf_threshold,
            "nms_threshold": self.meta.nms_threshold,
            "classes_of_interest": list(self.meta.classes_of_interest),
            "batch_size": self.meta.batch_size,
            "timestamp": self.meta.timestamp or datetime.now().isoformat(timespec="seconds"),
            "total_images": self.meta.total_images,
            "total_detections": self.meta.total_detections,
            "total_time_ms": round(self.meta.total_time_ms, 2),
        }
        images = []
        for r in self.images:
            images.append({
                "path": r.path,
                "output": r.output,
                "width": r.width,
                "height": r.height,
                "inference_ms": round(r.inference_ms, 2),
                "detections": r.detections,
            })
        return {"meta": meta, "images": images}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self._to_dict(), fp, ensure_ascii=False, indent=2)
        return path


# ---------- Хелпер: Detection -> dict ----------
def detection_to_dict(d) -> dict:
    """
    Принимает объект Detection (из main.py) и делает сериализуемый dict.
    Не импортирует Detection, чтобы модуль был переиспользуемым.
    """
    x, y, w, h = d.box
    return {
        "class_id": d.class_id,
        "class_name": d.class_name,
        "score": round(float(d.score), 4),
        "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "xyxy": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
    }