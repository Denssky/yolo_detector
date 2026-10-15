"""
Веб-просмотрщик результатов детекции.
Читает output/detections.json и показывает галерею + сводку.

Запуск:  python web_viewer.py
Открыть: http://127.0.0.1:5000
"""

import json
import os
from collections import defaultdict

from flask import Flask, render_template, send_from_directory, abort, jsonify


# Пути относительно проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
JSON_PATH = os.path.join(OUTPUT_DIR, "detections.json")


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


# ---------- Фильтры шаблона ----------

@app.template_filter("basename")
def _basename(path: str) -> str:
    """Возвращает имя файла без папки."""
    return os.path.basename(path)


# ---------- Вспомогательные ----------

def load_report() -> dict:
    """Читает JSON-отчёт. Возвращает пустую структуру, если файла нет."""
    if not os.path.exists(JSON_PATH):
        return {"meta": {"total_time_ms": 0}, "images": []}
    with open(JSON_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_summary(images):
    """Сводка: [(class_name, count, avg_score), ...] по убыванию count."""
    per_class = defaultdict(list)
    for img in images:
        for d in img["detections"]:
            per_class[d["class_name"]].append(d["score"])
    return sorted(
        [(name, len(scores), sum(scores) / len(scores))
         for name, scores in per_class.items()],
        key=lambda x: -x[1],
    )


# ---------- Маршруты ----------

@app.route("/")
def index():
    data = load_report()
    summary = build_summary(data["images"])
    total_dets = sum(len(img["detections"]) for img in data["images"])
    return render_template(
        "index.html",
        data=data,
        summary=summary,
        total_dets=total_dets,
    )


@app.route("/image/<path:filename>")
def image(filename: str):
    """
    Отдаёт аннотированное изображение из output/.
    basename() — защита от path traversal: пользователь не может
    запросить ../../etc/passwd.
    """
    filename = os.path.basename(filename)
    full = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(full):
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/data")
def api_data():
    """JSON-API: отдаёт содержимое output/detections.json как есть."""
    return jsonify(load_report())


if __name__ == "__main__":
    print(f"Читаем: {JSON_PATH}")
    print("Откройте: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)