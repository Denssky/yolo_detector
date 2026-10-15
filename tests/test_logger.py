"""
Тесты для logger.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger, setup_logging


def test_setup_logging_default():
    """По умолчанию уровень INFO."""
    setup_logging()
    logger = get_logger()
    assert logger.level == logging.INFO


def test_setup_logging_warning():
    """Явный уровень WARNING."""
    setup_logging(level="WARNING")
    logger = get_logger()
    assert logger.level == logging.WARNING


def test_setup_logging_debug():
    """DEBUG устанавливается корректно."""
    setup_logging(level="DEBUG")
    logger = get_logger()
    assert logger.level == logging.DEBUG


def test_setup_logging_unknown_level():
    """Неизвестный уровень → INFO (fallback)."""
    setup_logging(level="NOT_A_LEVEL")
    logger = get_logger()
    assert logger.level == logging.INFO


def test_setup_logging_log_file(tmp_path):
    """С log_file создаётся файл-хендлер."""
    log_path = tmp_path / "test.log"
    setup_logging(level="INFO", log_file=str(log_path))
    logger = get_logger()
    logger.info("test message")

    # Проверяем, что файл создан и содержит сообщение
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "test message" in content


def test_logger_singleton():
    """get_logger всегда возвращает один и тот же объект."""
    setup_logging()
    a = get_logger()
    b = get_logger()
    assert a is b