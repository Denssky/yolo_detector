"""
Тесты для io_utils.collect_images.
"""

import os
import sys

import cv2
import numpy as np
import pytest

# Добавляем корень проекта в sys.path, чтобы импортировать модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from io_utils import collect_images


@pytest.fixture
def sample_dir(tmp_path):
    """Создаёт временную папку с тестовыми изображениями."""
    d = tmp_path / "images"
    d.mkdir()
    # Три .jpg + один .png
    for i, name in enumerate(["a.jpg", "b.jpg", "c.png"]):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(d / name), img)
    # Мусорный файл — не изображение
    (d / "readme.txt").write_text("not an image")
    return d


def test_collect_from_directory(sample_dir):
    """Из папки собираются только файлы с известными расширениями."""
    files = collect_images([str(sample_dir)])
    names = sorted(os.path.basename(f) for f in files)
    assert names == ["a.jpg", "b.jpg", "c.png"]


def test_collect_single_file(sample_dir):
    """Отдельный файл принимается как есть."""
    path = str(sample_dir / "a.jpg")
    files = collect_images([path])
    assert files == [path]


def test_collect_glob(tmp_path, sample_dir):
    """Glob-паттерн раскрывается."""
    pattern = str(sample_dir / "*.jpg")
    files = collect_images([pattern])
    names = sorted(os.path.basename(f) for f in files)
    assert names == ["a.jpg", "b.jpg"]


def test_collect_missing_path(caplog):
    """Несуществующий путь — предупреждение в лог, но не падение."""
    import logging
    with caplog.at_level(logging.WARNING, logger="yolo_detector"):
        files = collect_images(["definitely_does_not_exist_12345.jpg"])
    assert files == []
    assert "не найдено" in caplog.text


def test_collect_deduplicates(tmp_path, sample_dir):
    """Один и тот же путь дважды — не дублируется."""
    p = str(sample_dir / "a.jpg")
    files = collect_images([p, p, str(sample_dir)])
    names = [os.path.basename(f) for f in files]
    # a.jpg должен быть один раз
    assert names.count("a.jpg") == 1


def test_collect_sorted(sample_dir):
    """Результат отсортирован."""
    files = collect_images([str(sample_dir)])
    assert files == sorted(files)