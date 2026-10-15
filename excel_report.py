"""
Excel-отчёт по результатам детекции.
Два листа: «Detections» (одна строка = одна детекция) и «Summary» (сводка по классам).
"""

import os
from collections import defaultdict
from typing import Optional

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter


HEADER_FILL = PatternFill(start_color="4C9AFF", end_color="4C9AFF", fill_type="solid")
HEADER_FONT = Font(bold=True, color="FFFFFF")
CENTER = Alignment(horizontal="center", vertical="center")


def _autosize_columns(ws, min_width: int = 8, max_width: int = 40) -> None:
    """Подгоняет ширину колонок по содержимому."""
    for col_idx, column in enumerate(ws.columns, start=1):
        max_len = 0
        for cell in column:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        width = min(max(min_width, max_len + 2), max_width)
        ws.column_dimensions[get_column_letter(col_idx)].width = width


def save_excel(report, path: str = "output/detections.xlsx") -> Optional[str]:
    """
    Сохраняет два листа:
      - Detections: file, class_name, class_id, score, x, y, w, h
      - Summary:    class_name, count, avg_score, min_score, max_score
    Возвращает путь к файлу или None, если детекций нет.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    wb = Workbook()

    # --- Лист 1: все детекции ---
    ws = wb.active
    ws.title = "Detections"
    headers = ["file", "class_name", "class_id", "score", "x", "y", "w", "h"]
    ws.append(headers)
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
    ws.freeze_panes = "A2"   # заголовок остаётся при прокрутке

    rows = 0
    for img in report.images:
        for d in img.detections:
            box = d["box"]
            ws.append([
                img.path,
                d["class_name"],
                d["class_id"],
                round(float(d["score"]), 4),
                box["x"], box["y"], box["w"], box["h"],
            ])
            rows += 1
    _autosize_columns(ws)

    # --- Лист 2: сводка по классам ---
    per_class_scores: dict = defaultdict(list)
    for img in report.images:
        for d in img.detections:
            per_class_scores[d["class_name"]].append(d["score"])

    ws2 = wb.create_sheet("Summary")
    ws2.append(["class_name", "count", "avg_score", "min_score", "max_score"])
    for cell in ws2[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
    ws2.freeze_panes = "A2"

    items = sorted(per_class_scores.items(),
                   key=lambda kv: (-len(kv[1]), kv[0]))
    for name, scores in items:
        n = len(scores)
        avg = sum(scores) / n
        ws2.append([name, n,
                    round(avg, 4),
                    round(min(scores), 4),
                    round(max(scores), 4)])
    _autosize_columns(ws2)

    wb.save(path)
    return path