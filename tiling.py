"""
Тайлинг больших изображений для детекции мелких объектов.

Идея: режем оригинал на плитки фиксированного размера с перекрытием,
детектим каждую отдельно, потом склеиваем координаты в систему оригинала
и применяем глобальный NMS.
"""

from typing import List, Tuple

import cv2
import numpy as np

from detector import Detection


def compute_tiles(width: int, height: int,
                  tile_size: int,
                  overlap: int) -> List[Tuple[int, int, int, int]]:
    """
    Возвращает список плиток (x1, y1, x2, y2) в системе оригинала.
    Плитки перекрываются на `overlap` пикселей.
    """
    if tile_size <= 0 or width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]

    step = tile_size - overlap
    if step <= 0:
        step = tile_size  # защита от бесконечного цикла

    tiles = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            x2 = min(x + tile_size, width)
            y2 = min(y + tile_size, height)
            # Последняя плитка может быть меньше — сдвигаем её назад,
            # чтобы она была полного размера (кроме совсем маленьких изображений).
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
            if x2 >= width:
                break
            x += step
        if y2 >= height:
            break
        y += step

    return tiles


def global_nms(detections: List[Detection],
               iou_threshold: float = 0.4) -> List[Detection]:
    """
    NMS на всём списке детекций (после склейки тайлов).
    Работает по классам: боксы разных классов не подавляют друг друга.
    """
    if not detections:
        return []

    by_class: dict = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    result: List[Detection] = []
    for cid, dets in by_class.items():
        boxes = [[d.x1, d.y1, d.box[2], d.box[3]] for d in dets]
        scores = [d.score for d in dets]
        idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_threshold)
        for i in np.array(idxs).flatten():
            result.append(dets[int(i)])
    return result