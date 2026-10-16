"""
Сервисные CLI-команды: --version, --check, --list-classes, --list-presets.
Не запускают детекцию — быстро выполняются и завершаются.
"""

import os
import sys
from importlib.metadata import version as pkg_version, PackageNotFoundError

import cv2


# ---------- --version ----------

def _lib_version(name: str) -> str:
    """Возвращает версию установленного пакета или '(не установлен)'."""
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "(не установлен)"


def print_version(version: str) -> None:
    """Печатает версию проекта и версии ключевых библиотек."""
    import numpy

    print(f"yolo_detector {version}")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  OpenCV:  {cv2.__version__}")
    print(f"  numpy:   {numpy.__version__}")
    print(f"  matplotlib: {_lib_version('matplotlib')}")
    print(f"  openpyxl:   {_lib_version('openpyxl')}")
    print(f"  flask:      {_lib_version('flask')}")


# ---------- --check ----------

def run_check(paths: dict) -> int:
    """
    Проверяет наличие моделей и зависимостей.
    paths = {
        "YOLO weights": "...",
        "YOLO config":  "...",
        "YOLO names":   "...",
        "YuNet model":  "...",
    }
    Возвращает 0, если всё на месте, 1 — если чего-то не хватает.
    """
    print("--- Проверка окружения ---")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")

    if not hasattr(cv2, "FaceDetectorYN"):
        print("  [!] cv2.FaceDetectorYN недоступен "
              "(нужна OpenCV 4.5.4+ с contrib)")

    build_info = cv2.getBuildInformation()
    print(f"FFmpeg в OpenCV: "
          f"{'да' if 'FFMPEG' in build_info else 'нет'}")

    print("\n--- Модели ---")
    all_ok = True
    for label, path in paths.items():
        exists = os.path.exists(path)
        mark = "OK " if exists else "!! "
        size = ""
        if exists:
            mb = os.path.getsize(path) / (1024 * 1024)
            size = f"  ({mb:.2f} MB)"
        print(f"  {mark}{label}: {path}{size}")
        if not exists:
            all_ok = False

    print()
    if all_ok:
        print("Все модели на месте.")
        return 0
    else:
        print("[!] Не все модели найдены.")
        print("Скачать — см. README, раздел «Установка».")
        return 1


# ---------- --list-classes ----------

def print_classes(names_path: str) -> int:
    """Печатает все классы COCO с индексами. Возвращает код возврата."""
    if not os.path.exists(names_path):
        print(f"[warn] не найден файл: {names_path}", file=sys.stderr)
        return 1

    with open(names_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    print(f"Классы COCO ({len(classes)}):")
    for i, name in enumerate(classes):
        print(f"  {i:>2}  {name}")
    return 0


# ---------- --list-presets ----------

def print_presets(presets: dict) -> int:
    """Печатает доступные пресеты классов и их состав."""
    print("Пресеты классов:")
    for name, classes in presets.items():
        if not classes:
            print(f"  {name:<10} (все 80 классов COCO)")
        else:
            print(f"  {name:<10} {len(classes):>2} класс(ов): "
                  f"{', '.join(classes)}")
    print("\nИспользование: python main.py images/ --preset <имя>")
    return 0