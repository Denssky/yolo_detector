"""
Тесты для config_loader.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_loader import load_config, merge_into_args


class Args:
    """Простой заменитель argparse.Namespace для тестов."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_load_config_missing():
    """Пустой путь — пустой dict."""
    assert load_config("") == {}


def test_load_config_not_found():
    """Несуществующий файл — FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("definitely_does_not_exist.json")


def test_load_config_valid(tmp_path):
    """Корректный JSON читается."""
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"conf": 0.5, "preset": "traffic"}))
    data = load_config(str(p))
    assert data["conf"] == 0.5
    assert data["preset"] == "traffic"


def test_load_config_not_object(tmp_path):
    """JSON-массив вместо объекта — ValueError."""
    p = tmp_path / "bad.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_merge_applies_config_when_cli_untouched():
    """CLI не передан — берётся значение из конфига."""
    defaults = Args(conf=0.4, preset=None, faces=False)
    args = Args(conf=0.4, preset=None, faces=False)
    config = {"conf": 0.2, "preset": "traffic", "faces": True}

    merged, applied = merge_into_args(args, config, defaults)

    assert merged.conf == 0.2
    assert merged.preset == "traffic"
    assert merged.faces is True
    assert set(applied) == {"conf", "preset", "faces"}


def test_merge_cli_wins():
    """CLI-флаг имеет приоритет над конфигом."""
    defaults = Args(conf=0.4, preset=None)
    args = Args(conf=0.9, preset=None)   # пользователь передал --conf 0.9
    config = {"conf": 0.2, "preset": "traffic"}

    merged, applied = merge_into_args(args, config, defaults)

    # conf не изменён (пользователь задал явно)
    assert merged.conf == 0.9
    # preset применён (не задан в CLI)
    assert merged.preset == "traffic"
    assert "preset" in applied
    assert "conf" not in applied


def test_merge_ignores_unknown_key(capsys):
    """Неизвестный ключ — предупреждение, не падение."""
    defaults = Args(conf=0.4)
    args = Args(conf=0.4)
    config = {"foo": "bar", "conf": 0.3}

    merged, applied = merge_into_args(args, config, defaults)

    assert merged.conf == 0.3
    captured = capsys.readouterr()
    assert "неизвестный ключ" in captured.out