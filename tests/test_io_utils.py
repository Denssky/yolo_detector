"""
Тесты для io_utils: save_crops, save_csv.
"""

import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import Detection
from io_utils import save_crops


def test_save_crops_basic(tmp_path):
    """Кропы сохраняются с правильными именами."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    dets = [
        Detection(box=(10, 10, 30, 30), class_id=7, score=0.9,
                  class_name="truck"),
        Detection(box=(50, 50, 20, 20), class_id=7, score=0.8,
                  class_name="truck"),
    ]
    counter = {}
    saved = save_crops(img, dets, "photo.jpg", str(tmp_path), counter)

    assert saved == 2
    files = sorted(os.listdir(tmp_path))
    assert files == ["photo_truck_001.jpg", "photo_truck_002.jpg"]


def test_save_crops_empty(tmp_path):
    """Нет детекций — нет файлов."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    saved = save_crops(img, [], "photo.jpg", str(tmp_path), {})
    assert saved == 0
    assert os.listdir(tmp_path) == []


def test_save_crops_out_of_bounds(tmp_path):
    """Бокс за границами изображения обрезается."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    dets = [Detection(box=(40, 40, 100, 100), class_id=7, score=0.9,
                      class_name="truck")]
    saved = save_crops(img, dets, "photo.jpg", str(tmp_path), {})
    assert saved == 1
    crop = cv2.imread(str(tmp_path / "photo_truck_001.jpg"))
    # Кроп не больше изображения
    assert crop.shape[0] <= 50
    assert crop.shape[1] <= 50


def test_save_crops_spaces_in_class(tmp_path):
    """Пробелы в имени класса заменяются на подчёркивания."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [Detection(box=(10, 10, 20, 20), class_id=0, score=0.9,
                      class_name="cell phone")]
    save_crops(img, dets, "img.jpg", str(tmp_path), {})
    files = os.listdir(tmp_path)
    assert files == ["img_cell_phone_001.jpg"]