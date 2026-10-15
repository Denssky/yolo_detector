"""
Централизованное логирование.
Прогресс и служебные сообщения — в stderr через logging.
Итоговые результаты — в stdout через print.
"""

import logging
import os
import sys
from typing import Optional


_LOGGER_NAME = "yolo_detector"


def get_logger() -> logging.Logger:
    """Возвращает логгер проекта."""
    return logging.getLogger(_LOGGER_NAME)


def setup_logging(level: str = "INFO",
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Настраивает логгер: stderr для прогресса, опционально файл для полного лога.
    Вызывается один раз в начале parse_args().
    """
    logger = get_logger()
    logger.propagate = False

    # Чистим старые хендлеры (на случай повторного вызова)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # --- Консоль: короткий формат, в stderr ---
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(
        fmt="[%(levelname)s] %(message)s"
    ))
    logger.addHandler(console)

    # --- Файл: полный формат с датой и временем ---
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger