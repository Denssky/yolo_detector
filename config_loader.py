"""
Загрузка настроек из JSON-файла (settings.json).
CLI-аргументы имеют приоритет над значениями из конфига.
"""

import json
import os
from typing import Any, Dict, Tuple


# Сопоставление имён из JSON с dest-именами argparse.
CONFIG_KEY_MAP = {
    "preset":       "preset",
    "classes":      "classes",
    "conf":         "conf",
    "nms":          "nms",
    "batch_size":   "batch_size",
    "output_dir":   "output_dir",
    "json":         "json",
    "csv":          "csv",
    "xlsx":         "xlsx",
    "save_crops":   "save_crops",
    "crops_dir":    "crops_dir",
    "faces":        "faces",
    "face_model":   "face_model",
    "face_conf":    "face_conf",
    "plots":        "plots",
    "plots_path":   "plots_path",
    "show":         "show",
    "verbose":      "verbose",
    "cuda":         "cuda",
    "inputs":       "inputs",
    "video":         "video",
    "webcam":        "webcam",
    "save_video":    "save_video",
    "save_every":    "save_every",
    "infer_every":   "infer_every",
    "max_frames":    "max_frames",
    
}


def load_config(path: str) -> Dict[str, Any]:
    """Читает JSON-конфиг. Возвращает пустой dict, если path пустой."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Конфиг должен быть JSON-объектом: {path}")
    return data


def merge_into_args(args, config: Dict[str, Any], defaults) -> Tuple[Any, list]:
    """
    Мержит config в args.
    Правило: если пользователь явно передал флаг в CLI — CLI побеждает.
    Иначе — берётся значение из config (если есть).

    defaults — Namespace с дефолтными значениями (parse_args([])).
    Возвращает (args, список_применённых_ключей).
    """
    applied = []
    for json_key, value in config.items():
        dest = CONFIG_KEY_MAP.get(json_key)
        if dest is None:
            print(f"[config] неизвестный ключ: {json_key!r} — пропущен")
            continue
        if not hasattr(args, dest):
            print(f"[config] ключ {json_key!r} → нет dest {dest!r} — пропущен")
            continue

        current = getattr(args, dest)
        default = getattr(defaults, dest, None)

        # Если пользователь не менял (значение равно дефолту) — применяем.
        if current == default:
            setattr(args, dest, value)
            applied.append(json_key)

    return args, applied