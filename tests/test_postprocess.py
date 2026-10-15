"""
Тесты для YoloDetector._postprocess.
Проверяют корректность разбора выходов сети на синтетических данных.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import YoloDetector


# Заглушка: создаём YoloDetector, но без загрузки реальной модели.
# Проверяем только _postprocess — он не использует self.net.
@pytest.fixture
def detector():
    obj = YoloDetector.__new__(YoloDetector)
    obj.conf_threshold = 0.5
    obj.nms_threshold = 0.4
    obj.classes_of_interest = {"truck"}
    # 80 классов COCO с truck на позиции 7
    obj.classes = ["person"] * 80
    obj.classes[7] = "truck"
    return obj


def make_out(n_rows: int, n_cols: int = 85, tx: float = 0.5, ty: float = 0.5,
             tw: float = 0.1, th: float = 0.1, obj_conf: float = 0.9,
             class_id: int = 7, class_conf: float = 0.9) -> np.ndarray:
    """
    Формирует выход сети формы (n_rows, 85).
    Полезная детекция в первой строке, остальные — нули.
    """
    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    out[0, 0] = tx
    out[0, 1] = ty
    out[0, 2] = tw
    out[0, 3] = th
    out[0, 4] = obj_conf
    out[0, 5 + class_id] = class_conf
    return out


def test_empty_outs(detector):
    """Пустой список выходов — пустой результат."""
    result = detector._postprocess([], [(480, 640)])
    assert result == [[]]


def test_single_detection_found(detector):
    """Одна уверенная детекция truck находится."""
    out = make_out(n_rows=100, class_id=7, class_conf=0.9)
    result = detector._postprocess([out], [(480, 640)])
    assert len(result) == 1
    dets = result[0]
    assert len(dets) == 1
    assert dets[0].class_name == "truck"
    assert dets[0].score > 0.8


def test_low_confidence_filtered(detector):
    """Детекция ниже порога отбрасывается."""
    out = make_out(n_rows=100, class_conf=0.3)   # порог 0.5
    result = detector._postprocess([out], [(480, 640)])
    assert result == [[]]


def test_wrong_class_filtered(detector):
    """Класс не из classes_of_interest отбрасывается."""
    out = make_out(n_rows=100, class_id=0, class_conf=0.9)   # person
    result = detector._postprocess([out], [(480, 640)])
    assert result == [[]]


def test_batch_size_2(detector):
    """Батч из двух изображений обрабатывается корректно."""
    out = np.zeros((200, 85), dtype=np.float32)
    # Для изображения 0 — truck в строке 0
    out[0, :] = [0.5, 0.5, 0.1, 0.1, 0.9, *([0.0] * 80)]
    out[0, 5 + 7] = 0.9
    # Для изображения 1 — truck в строке 100
    out[100, :] = [0.5, 0.5, 0.1, 0.1, 0.9, *([0.0] * 80)]
    out[100, 5 + 7] = 0.9

    result = detector._postprocess([out], [(480, 640), (480, 640)])
    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 1
    assert result[0][0].class_name == "truck"
    assert result[1][0].class_name == "truck"


def test_transposed_output(detector):
    """Выход в форме (85, N) транспонируется и обрабатывается."""
    out = make_out(n_rows=100).T   # (85, 100)
    result = detector._postprocess([out], [(480, 640)])
    assert len(result) == 1
    assert len(result[0]) == 1


def test_box_coordinates_scaling(detector):
    """Координаты пересчитываются в пиксели исходного изображения."""
    # Центр (0.5, 0.5), размер (0.2, 0.2) — для 640x480:
    # cx=320, cy=240, w=128, h=96 → x1=256, y1=192
    out = make_out(n_rows=100, tx=0.5, ty=0.5, tw=0.2, th=0.2,
                   class_id=7, class_conf=0.9)
    result = detector._postprocess([out], [(480, 640)])
    d = result[0][0]
    assert d.x1 == 256
    assert d.y1 == 192
    assert d.box[2] == 128
    assert d.box[3] == 96