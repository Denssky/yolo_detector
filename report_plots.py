"""
Графики результатов детекции.
Читает ReportBuilder и сохраняет PNG со сводкой.
"""

import os
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless-бэкенд — сохраняет в файл без GUI
import matplotlib.pyplot as plt


def save_report_plots(report, path: str = "output/report.png") -> Optional[str]:
    """
    Строит и сохраняет 4 подграфика по данным report.images.
    Возвращает путь к PNG или None, если детекций нет.
    """
    per_class_count: dict = defaultdict(int)
    per_class_scores: dict = defaultdict(list)
    all_scores: list = []
    box_sizes: list = []

    for img in report.images:
        for d in img.detections:
            cls = d["class_name"]
            per_class_count[cls] += 1
            per_class_scores[cls].append(d["score"])
            all_scores.append(d["score"])
            box_sizes.append((d["box"]["w"], d["box"]["h"]))

    if not all_scores:
        print("[plots] нет детекций — график не строится")
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Detection Report", fontsize=16, fontweight="bold")

    # 1. Количество по классам
    ax = axes[0, 0]
    classes = sorted(per_class_count.keys(),
                     key=lambda c: -per_class_count[c])
    counts = [per_class_count[c] for c in classes]
    ax.barh(classes[::-1], counts[::-1], color="#4C9AFF")
    ax.set_xlabel("Количество")
    ax.set_title("Объекты по классам")
    ax.grid(axis="x", alpha=0.3)

    # 2. Гистограмма score
    ax = axes[0, 1]
    ax.hist(all_scores, bins=20, color="#36B37E", edgecolor="white")
    ax.set_xlabel("Score")
    ax.set_ylabel("Количество")
    ax.set_title(f"Распределение score (n={len(all_scores)})")
    ax.grid(alpha=0.3)

    # 3. Scatter размеров боксов
    ax = axes[1, 0]
    if box_sizes:
        xs = [w for w, _ in box_sizes]
        ys = [h for _, h in box_sizes]
        ax.scatter(xs, ys, alpha=0.6, color="#FF5630")
    ax.set_xlabel("Ширина бокса, px")
    ax.set_ylabel("Высота бокса, px")
    ax.set_title("Размеры боксов")
    ax.grid(alpha=0.3)

    # 4. Средний score по классам
    ax = axes[1, 1]
    avg_scores = [sum(per_class_scores[c]) / len(per_class_scores[c])
                  for c in classes]
    ax.barh(classes[::-1], avg_scores[::-1], color="#FFAB00")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Средний score")
    ax.set_title("Уверенность по классам")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path