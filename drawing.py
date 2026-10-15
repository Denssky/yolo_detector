"""
Отрисовка результатов детекции и интерактивный просмотр.
"""

from typing import List, Tuple

import cv2
import numpy as np

from detector import Detection
from face_detector import FaceDetection


# ---------- Цвета (BGR, как в OpenCV) ----------

TEXT_COLOR  = (0, 0, 0)         # подписи на цветных плашках
COUNT_COLOR = (0, 0, 255)       # общий счётчик сверху
DEFAULT_BOX_COLOR = (0, 255, 0) # fallback
FACE_COLOR = (0, 255, 255)      # жёлтый — боксы лиц
FACE_LANDMARK_COLOR = (0, 0, 255)  # красный — ключевые точки

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
    """Стабильный цвет для класса: одинаковый между запусками."""
    return CLASS_PALETTE[class_id % len(CLASS_PALETTE)]


# ---------- Отрисовка ----------

def draw_detections(image: np.ndarray,
                    detections: List[Detection]) -> np.ndarray:
    """Рисует боксы YOLO-детекций и счётчик."""
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


def draw_faces(image: np.ndarray,
               faces: List[FaceDetection]) -> np.ndarray:
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


# ---------- Интерактивный просмотр ----------

def browse_results(results: List[Tuple[str, np.ndarray]],
                   window_name: str = "YOLOv4-tiny") -> None:
    """
    Пролистывание аннотированных изображений.
    Управление:
      D / Space / ] — следующее
      A / [         — предыдущее
      Q / Esc       — выход
    """
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