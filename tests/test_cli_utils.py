"""
Тесты для cli_utils: print_version, print_classes, print_presets.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli_utils import print_classes, print_presets, print_version


def test_print_version(capsys):
    """--version печатает строку версии и версии библиотек."""
    print_version("9.9.9")
    out = capsys.readouterr().out
    assert "yolo_detector 9.9.9" in out
    assert "Python:" in out
    assert "OpenCV:" in out


def test_print_classes_missing_file(capsys):
    """Несуществующий файл — код возврата 1."""
    code = print_classes("definitely_does_not_exist.names")
    assert code == 1


def test_print_classes_ok(tmp_path, capsys):
    """Существующий файл — классы печатаются с индексами."""
    p = tmp_path / "coco.names"
    p.write_text("person\ncar\ntruck\n")
    code = print_classes(str(p))
    out = capsys.readouterr().out
    assert code == 0
    assert "person" in out
    assert "car" in out
    assert "truck" in out
    assert "(3)" in out   # количество классов


def test_print_presets(capsys):
    """Пресеты печатаются с составом."""
    presets = {
        "truck": ("truck",),
        "all": (),
    }
    print_presets(presets)
    out = capsys.readouterr().out
    assert "truck" in out
    assert "все 80 классов COCO" in out