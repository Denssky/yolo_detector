# main.py — детектор объектов на изображениях через YOLOv4-tiny и OpenCV DNN.
#
# Возможности:
#   - обработка файлов, папок и glob-паттернов;
#   - фильтр по классам COCO (через --classes или готовые --preset);
#   - батч-инференс с автоматической группировкой по размеру изображения;
#   - сохранение аннотированных картинок, сводного JSON, CSV и кропов объектов;
#   - интерактивный просмотр результатов в окне OpenCV.

import os
import sys
import time
import glob
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Sequence
from collections import defaultdict

import cv2
import numpy as np

# Локальный модуль отчёта — создаёт JSON со всеми детекциями.
# Держим его отдельно, чтобы main.py не разрастался.
from json_report import ReportBuilder, ReportMeta, ImageRecord, detection_to_dict


# ============================================================================
# Константы
# ============================================================================

# Пути к файлам модели YOLOv4-tiny (веса, конфиг, имена классов COCO).
MODEL_DIR = "yolo_tiny"
WEIGHTS   = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
CONFIG    = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
NAMES     = os.path.join(MODEL_DIR, "coco.names")

# Входной размер сети. 416x416 — стандарт для YOLOv4-tiny.
# Меньше (320) — быстрее, но хуже ловит мелкие объекты.
INPUT_SIZE      = (416, 416)

# Порог уверенности: детекции ниже отбрасываются.
DEFAULT_CONF    = 0.4

# Порог NMS (Non-Maximum Suppression): подавляет перекрывающиеся боксы.
DEFAULT_NMS     = 0.4

# Классы по умолчанию — только грузовики.
DEFAULT_CLASSES = ("truck",)

# Готовые наборы классов для --preset. Пустой кортеж = все 80 классов COCO.
CLASS_PRESETS = {
    "truck":   ("truck",),
    "traffic": ("person", "car", "bus", "truck", "motorcycle", "bicycle",
                "traffic light", "stop sign"),
    "office":  ("person", "chair", "laptop", "keyboard", "mouse",
                "cell phone", "book", "clock", "cup"),
    "food":    ("banana", "apple", "sandwich", "orange", "pizza",
                "donut", "cake", "bottle", "cup", "bowl", "fork", "knife"),
    "all":     (),
}

# Цвета для отрисовки. BGR — порядок каналов в OpenCV, а не RGB.
TEXT_COLOR  = (0, 0, 0)        # чёрный — для подписей на цветных плашках
COUNT_COLOR = (0, 0, 255)      # красный — для общего счётчика сверху
DEFAULT_BOX_COLOR = (0, 255, 0)  # зелёный — fallback (сейчас не используется)

# Палитра из 12 различимых цветов. Класс получает цвет по формуле
# class_id % len(CLASS_PALETTE), то есть один и тот же класс всегда
# рисуется одним цветом — независимо от порядка появления на кадре.
CLASS_PALETTE = [
    (0, 255, 0),      # зелёный
    (0, 165, 255),    # оранжевый
    (255, 0, 0),      # синий
    (0, 0, 255),      # красный
    (255, 255, 0),    # голубой
    (255, 0, 255),    # пурпурный
    (0, 255, 255),    # жёлтый
    (128, 0, 128),    # фиолетовый
    (0, 128, 255),    # тёмно-оранжевый
    (128, 128, 0),    # тёмно-голубой
    (0, 128, 0),      # тёмно-зелёный
    (203, 192, 255),  # розовый
]


def color_for_class(class_id: int) -> tuple:
    """Стабильный цвет для класса. Одинаковый между запусками."""
    return CLASS_PALETTE[class_id % len(CLASS_PALETTE)]


# ============================================================================
# Модель данных
# ============================================================================

@dataclass
class Detection:
    """
    Одна детекция: бокс, класс, уверенность.
    box — (x, y, w, h), где (x, y) — левый верхний угол.
    Удобные свойства x1/y1/x2/y2 для отрисовки и обрезки кропов.
    """
    box: Tuple[int, int, int, int]   # x, y, w, h
    class_id: int
    score: float
    class_name: str = ""

    @property
    def x1(self): return self.box[0]
    @property
    def y1(self): return self.box[1]
    @property
    def x2(self): return self.box[0] + self.box[2]
    @property
    def y2(self): return self.box[1] + self.box[3]


# ============================================================================
# Детектор
# ============================================================================

class YoloDetector:
    """
    Обёртка над cv2.dnn для YOLO.
    Держит загруженную сеть, пороги, список классов и имена выходных слоёв.
    """

    def __init__(self, weights: str, config: str, names: str,
                 input_size=INPUT_SIZE,
                 conf_threshold=DEFAULT_CONF,
                 nms_threshold=DEFAULT_NMS,
                 classes_of_interest=DEFAULT_CLASSES,
                 use_cuda: bool = False):
        # Проверяем, что все файлы модели на месте — иначе понятная ошибка
        for p in (weights, config, names):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Не найден файл: {p}")

        # Загружаем сеть из .weights + .cfg
        self.net = cv2.dnn.readNet(weights, config)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        # Множество для быстрой проверки "интересен ли класс" через `in`
        self.classes_of_interest = set(classes_of_interest)

        # Backend: CUDA, если попросили и если сборка OpenCV её видит.
        # Иначе — CPU (DNN_BACKEND_OPENCV + DNN_TARGET_CPU).
        if use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # У YOLOv4-tiny два выходных слоя (сетки 13x13 и 26x26).
        # getUnconnectedOutLayers возвращает индексы с 1, поэтому -1.
        layer_names = self.net.getLayerNames()
        self.out_layers = [layer_names[i - 1]
                           for i in self.net.getUnconnectedOutLayers()]

        # Читаем имена классов COCO, по одному на строку.
        # strip() убирает \r, пробелы и BOM в начале файла.
        with open(names, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f if line.strip()]

    def _filter_class_ids(self) -> set:
        """
        ID классов, которые нас интересуют.
        Если список пуст (--preset all), возвращаем все ID — фильтр отключён.
        """
        if not self.classes_of_interest:
            return set(range(len(self.classes)))
        return {i for i, name in enumerate(self.classes)
                if name in self.classes_of_interest}

    def _postprocess(self, outs,
                     orig_sizes: Sequence[Tuple[int, int]]
                     ) -> List[List[Detection]]:
        """
        Преобразует сырые выходы сети в список Detection по каждому изображению.

        outs: список выходов сети. Каждый out имеет форму (N, 85), где
              N = batch_size * anchors. 85 = 4 координаты + 1 objectness +
              80 классов COCO.
        orig_sizes: [(height, width), ...] — исходные размеры изображений
                    батча. Нужны, чтобы перевести нормализованные координаты
                    (0..1) в пиксели конкретного изображения.

        Возвращает: results[b] — список Detection для b-го изображения батча.
        """
        batch_size = len(orig_sizes)

        # Аккумуляторы: для каждого изображения батча копим боксы/score/class_id.
        per_image_boxes: List[List[List[int]]] = [[] for _ in range(batch_size)]
        per_image_scores: List[List[float]] = [[] for _ in range(batch_size)]
        per_image_class_ids: List[List[int]] = [[] for _ in range(batch_size)]

        for out in outs:
            # --- Нормализация формы выхода к (N, 85) ---
            # Разные сборки OpenCV могут отдавать выход в разных формах.
            # Приводим к каноническому виду, иначе пропускаем.
            if out.ndim != 2:
                out = out.reshape(out.shape[0], -1)
            if out.shape[1] != 85 and out.shape[0] == 85:
                out = out.T
            if out.shape[1] != 85:
                continue

            rows = out.shape[0]
            # Строк должно быть ровно batch_size * anchors.
            # Если не делится — что-то не так, пропускаем.
            if rows % batch_size != 0:
                continue
            anchors = rows // batch_size

            for b in range(batch_size):
                h, w = orig_sizes[b]
                # Вырезаем часть выхода, относящуюся к b-му изображению.
                chunk = out[b * anchors:(b + 1) * anchors]

                # --- Векторизованная фильтрация по уверенности ---
                # Столбцы 5: — это 80 классов. Берём максимальный
                # по каждому ряду и его индекс.
                class_scores = chunk[:, 5:]
                class_ids = np.argmax(class_scores, axis=1)
                scores = class_scores[np.arange(len(class_ids)), class_ids]
                # Оставляем только те ряды, где score >= порога.
                mask = scores >= self.conf_threshold
                if not mask.any():
                    continue

                det = chunk[mask]
                cls = class_ids[mask]
                sc = scores[mask]

                # --- Координаты: из нормализованных (0..1) в пиксели ---
                # Формат YOLO: cx, cy, bw, bh — центр и размеры.
                cx = det[:, 0] * w
                cy = det[:, 1] * h
                bw = det[:, 2] * w
                bh = det[:, 3] * h
                # Переводим в левый верхний угол, обрезаем по 0.
                x = np.maximum(0, cx - bw / 2).astype(int)
                y = np.maximum(0, cy - bh / 2).astype(int)
                bw = bw.astype(int)
                bh = bh.astype(int)

                per_image_boxes[b].extend(
                    np.stack([x, y, bw, bh], axis=1).tolist())
                per_image_scores[b].extend(sc.tolist())
                per_image_class_ids[b].extend(cls.tolist())

        # --- NMS и фильтр по интересующим классам, для каждого изображения ---
        wanted = self._filter_class_ids()
        results: List[List[Detection]] = []
        for b in range(batch_size):
            boxes = per_image_boxes[b]
            scores = per_image_scores[b]
            class_ids = per_image_class_ids[b]
            if not boxes:
                results.append([])
                continue

            # NMS убирает дублирующиеся боксы (один объект — один бокс).
            idxs = cv2.dnn.NMSBoxes(boxes, scores,
                                    self.conf_threshold, self.nms_threshold)
            if len(idxs) == 0:
                results.append([])
                continue

            dets = []
            for i in np.array(idxs).flatten():
                i = int(i)
                cid = class_ids[i]
                # Фильтр по классам — только те, что в classes_of_interest.
                if cid not in wanted:
                    continue
                dets.append(Detection(
                    box=tuple(boxes[i]),
                    class_id=cid,
                    score=scores[i],
                    class_name=(self.classes[cid]
                                if cid < len(self.classes) else "unknown"),
                ))
            results.append(dets)
        return results

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Инференс на одном изображении. Возвращает список Detection."""
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]
                     ) -> List[List[Detection]]:
        """
        Инференс на батче изображений ОДНОГО размера.
        Возвращает список списков Detection — по одному на изображение.
        """
        if not images:
            return []
        orig_sizes = [img.shape[:2] for img in images]
        # blobFromImages: нормализует 0..1, меняет BGR->RGB (swapRB),
        # ресайзит к input_size. Требует одинаковых размеров картинок.
        blob = cv2.dnn.blobFromImages(
            list(images), 1 / 255.0, self.input_size, (0, 0, 0),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)
        return self._postprocess(outs, orig_sizes)


# ============================================================================
# Отрисовка
# ============================================================================

def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """
    Рисует боксы, подписи и счётчик на копии изображения.
    Изображение меняется in-place, возвращается тот же объект.
    """
    for d in detections:
        color = color_for_class(d.class_id)
        # Рамка вокруг объекта
        cv2.rectangle(image, (d.x1, d.y1), (d.x2, d.y2), color, 2)

        # Подпись "class score" с цветным фоном (для читаемости)
        label = f"{d.class_name} {d.score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (d.x1, d.y1 - th - base),
                      (d.x1 + tw, d.y1), color, -1)
        cv2.putText(image, label, (d.x1, d.y1 - base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

    # Общий счётчик детекций в левом верхнем углу
    cv2.putText(image, f"Detections: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COUNT_COLOR, 2, cv2.LINE_AA)
    return image


# ============================================================================
# Сохранение результатов
# ============================================================================

def save_crops(image: np.ndarray,
               detections: List[Detection],
               src_name: str,
               crops_dir: str,
               counter: dict) -> int:
    """
    Сохраняет вырезанные объекты в crops_dir.
    Имя файла: {stem}_{class_slug}_{номер:03d}.jpg
    counter — словарь {(stem, class_slug): int} для сквозной нумерации
              по каждому изображению и классу.
    Возвращает количество сохранённых кропов.
    """
    if not detections:
        return 0

    os.makedirs(crops_dir, exist_ok=True)
    stem = os.path.splitext(src_name)[0]
    h_img, w_img = image.shape[:2]
    saved = 0

    for d in detections:
        # Обрезаем бокс по границам изображения (детектор может выйти за край).
        x1 = max(0, d.x1)
        y1 = max(0, d.y1)
        x2 = min(w_img, d.x2)
        y2 = min(h_img, d.y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Пробелы в имени класса заменяем на подчёркивания:
        # "cell phone" -> "cell_phone" — иначе пробел в имени файла.
        class_slug = d.class_name.replace(" ", "_")
        key = (stem, class_slug)
        counter[key] = counter.get(key, 0) + 1
        num = counter[key]

        fname = f"{stem}_{class_slug}_{num:03d}.jpg"
        out_path = os.path.join(crops_dir, fname)
        cv2.imwrite(out_path, crop)
        saved += 1

    return saved


def save_csv(report: ReportBuilder, path: str) -> int:
    """
    Сохраняет плоскую таблицу детекций (одна строка = одна детекция).
    Колонки: file, class_name, class_id, score, x, y, w, h
    Возвращает число строк (без заголовка).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "class_name", "class_id",
                         "score", "x", "y", "w", "h"])
        for img in report.images:
            for d in img.detections:
                box = d["box"]
                writer.writerow([
                    img.path,
                    d["class_name"],
                    d["class_id"],
                    d["score"],
                    box["x"], box["y"], box["w"], box["h"],
                ])
                rows += 1
    return rows


def print_class_summary(report: ReportBuilder) -> None:
    """
    Печатает сводку по классам: количество, средний score, min/max.
    Читает report.images (ImageRecord с detections в виде dict).
    """
    stats: dict = {}   # class_name -> список score
    for img in report.images:
        for d in img.detections:
            stats.setdefault(d["class_name"], []).append(d["score"])

    print("\n--- Сводка по классам ---")
    if not stats:
        print("  (ничего не найдено)")
        return

    # Сортируем по убыванию количества, при равенстве — по имени.
    items = sorted(stats.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    name_w = max(len(name) for name, _ in items)
    total = 0
    for name, scores in items:
        n = len(scores)
        total += n
        avg = sum(scores) / n
        lo = min(scores)
        hi = max(scores)
        print(f"  {name:<{name_w}} : {n:>3}  "
              f"avg={avg:.2f}  ({lo:.2f} .. {hi:.2f})")
    print(f"Всего: {total} детекций, {len(items)} класс(ов)")


# ============================================================================
# Сбор файлов и CLI
# ============================================================================

def collect_images(paths: Iterable[str]) -> List[str]:
    """
    Принимает пути к файлам, папкам и glob-паттернам.
    Возвращает отсортированный список существующих файлов-изображений
    без дубликатов.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    found: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            # Папка — собираем все файлы с известными расширениями.
            for ext in exts:
                found.extend(glob.glob(os.path.join(p, f"*{ext}")))
                found.extend(glob.glob(os.path.join(p, f"*{ext.upper()}")))
        elif any(ch in p for ch in "*?["):
            # Glob-паттерн — раскрываем.
            found.extend(glob.glob(p, recursive=True))
        elif os.path.isfile(p):
            # Конкретный файл.
            found.append(p)
        else:
            print(f"[warn] не найдено: {p}", file=sys.stderr)

    # Дедупликация с сохранением порядка + сортировка для детерминизма.
    seen = set()
    unique = []
    for f in sorted(found):
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def parse_args():
    """Разбор аргументов командной строки."""
    p = argparse.ArgumentParser(description="YOLOv4-tiny object detector (images)")
    p.add_argument("inputs", nargs="*", default=["images/trucks.jpg"],
                   help="файлы, папки или glob-паттерны "
                        "(по умолчанию: images/trucks.jpg)")
    p.add_argument("--output-dir", default="output",
                   help="куда сохранять аннотированные изображения")
    p.add_argument("--json", default="output/detections.json",
                   help="путь к сводному JSON")
    p.add_argument("--csv", default="output/detections.csv",
                   help="путь к CSV со всеми детекциями "
                        "(пустая строка — не сохранять)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF,
                   help="порог уверенности (по умолчанию 0.4)")
    p.add_argument("--nms", type=float, default=DEFAULT_NMS,
                   help="порог NMS (по умолчанию 0.4)")
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES),
                   help="список классов COCO (имена с пробелами — в кавычках)")
    p.add_argument("--preset", choices=list(CLASS_PRESETS.keys()),
                   help="готовый набор классов (перебивает --classes)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="размер батча при инференсе")
    p.add_argument("--cuda", action="store_true",
                   help="использовать CUDA, если доступна")
    p.add_argument("--show", action="store_true",
                   help="открыть окно с результатом (по умолчанию — нет)")
    p.add_argument("--verbose", action="store_true",
                   help="печатать каждую детекцию")
    p.add_argument("--save-crops", action="store_true",
                   help="сохранять вырезанные объекты в <output-dir>/crops")
    p.add_argument("--crops-dir", default=None,
                   help="папка для кропов (по умолчанию <output-dir>/crops)")
    return p.parse_args()


def chunked(seq, n):
    """Разбивает последовательность на куски по n элементов."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ============================================================================
# Интерактивный просмотр
# ============================================================================

def browse_results(results: List[Tuple[str, np.ndarray]],
                   window_name: str = "YOLOv4-tiny") -> None:
    """
    Пролистывание аннотированных изображений.
    Управление:
      D / Space / ] — следующее
      A / [         — предыдущее
      Q / Esc       — выход
    """
    n = len(results)
    idx = 0
    print(f"\nПросмотр {n} изображений. "
          f"A/[ — назад, D/]/Space — вперёд, Q — выход.")

    while True:
        name, img = results[idx]
        display = img.copy()   # копия, чтобы не портить оригинал в списке

        # Подпись с номером и именем файла внизу — с двойной обводкой,
        # чтобы читалось на любом фоне (сначала чёрный толстый, поверх белый).
        hud = f"[{idx + 1}/{n}]  {name}"
        y = display.shape[0] - 15
        cv2.putText(display, hud, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, hud, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        # waitKey(0) — блокирует до любого нажатия.
        # Возвращает int, для ASCII-букв младший байт — сам код.
        key = cv2.waitKey(0)
        if key == -1:
            continue
        low = key & 0xFF

        if low in (ord("q"), ord("Q"), 27):        # Q/q/Esc — выход
            break
        if low in (ord("d"), ord("D"), 32, ord("]")):    # вперёд
            idx = (idx + 1) % n
        elif low in (ord("a"), ord("A"), ord("[")):      # назад
            idx = (idx - 1) % n

    cv2.destroyAllWindows()


# ============================================================================
# Точка входа
# ============================================================================

def main():
    args = parse_args()

    # Выбор классов: пресет приоритетнее, иначе — список из --classes.
    if args.preset:
        classes = CLASS_PRESETS[args.preset]
    else:
        classes = tuple(args.classes)

    # Создаём детектор. Модель загружается один раз — это ~200 мс.
    detector = YoloDetector(
        WEIGHTS, CONFIG, NAMES,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        classes_of_interest=classes,
        use_cuda=args.cuda,
    )

    # Собираем список файлов из аргументов (папки, glob, отдельные файлы).
    files = collect_images(args.inputs)
    if not files:
        print("Не найдено ни одного изображения.", file=sys.stderr)
        sys.exit(1)

    print(f"Классов: {len(detector.classes)}  |  файлов: {len(files)}")
    print(f"conf={args.conf}  nms={args.nms}  batch_size={args.batch_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Папка для кропов и счётчик (если включено).
    crops_dir = args.crops_dir or os.path.join(args.output_dir, "crops")
    crops_counter: dict = {}
    total_crops = 0
    if args.save_crops:
        os.makedirs(crops_dir, exist_ok=True)
        print(f"Кропы: {crops_dir}")

    # Сборщик JSON-отчёта.
    report = ReportBuilder(meta=ReportMeta(
        model=os.path.basename(MODEL_DIR),
        input_size=INPUT_SIZE,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        classes_of_interest=tuple(classes),
        batch_size=args.batch_size,
    ))

    total_time = 0.0
    saved_results: List[Tuple[str, np.ndarray]] = []

    # --- Загружаем все изображения один раз, группируем по размеру ---
    # blobFromImages требует одинаковых размеров в батче, поэтому
    # разные размеры обрабатываем отдельными батчами.
    by_size: dict = defaultdict(list)   # (h, w) -> [(path, img), ...]
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"[warn] не удалось прочитать: {f}", file=sys.stderr)
            continue
        by_size[img.shape[:2]].append((f, img))

    if not by_size:
        print("Не удалось прочитать ни одного изображения.", file=sys.stderr)
        sys.exit(1)

    n_groups = len(by_size)
    sizes_info = ", ".join(f"{h}x{w}: {len(v)}"
                           for (h, w), v in by_size.items())
    print(f"Групп по размеру: {n_groups} ({sizes_info})")

    # --- Основной цикл: по группам размеров, внутри — по батчам ---
    batch_idx = 0
    for (grp_h, grp_w), items in by_size.items():
        for batch_files in chunked(items, args.batch_size):
            batch_idx += 1
            valid_files = [f for f, _ in batch_files]
            images = [img for _, img in batch_files]

            t0 = time.perf_counter()
            batch_dets = detector.detect_batch(images)
            dt_batch = (time.perf_counter() - t0) * 1000
            total_time += dt_batch
            per_img_ms = dt_batch / len(images)

            print(f"[batch {batch_idx}] group {grp_h}x{grp_w}  "
                  f"{len(images)} img  "
                  f"{dt_batch:.1f} ms total  ({per_img_ms:.1f} ms/img)")

            # По каждому изображению батча: сохранить, при желании — кропы,
            # добавить запись в JSON-отчёт.
            for f, img, dets in zip(valid_files, images, batch_dets):
                h, w = img.shape[:2]
                name = os.path.basename(f)
                print(f"  {name:<30} найдено: {len(dets)}")

                if args.verbose:
                    for d in sorted(dets, key=lambda x: -x.score):
                        print(f"      {d.class_name:<12} "
                              f"score={d.score:.3f} box={d.box}")

                # Аннотированное изображение
                result = draw_detections(img, dets)
                out_path = os.path.join(args.output_dir, name)
                cv2.imwrite(out_path, result)
                saved_results.append((name, result))

                # Кропы объектов
                if args.save_crops:
                    total_crops += save_crops(img, dets, name,
                                              crops_dir, crops_counter)

                # Запись в JSON-отчёт
                report.add_image(ImageRecord(
                    path=f,
                    output=out_path,
                    width=w,
                    height=h,
                    inference_ms=per_img_ms,
                    detections=[detection_to_dict(d) for d in dets],
                ))

    if args.save_crops:
        print(f"Кропов сохранено: {total_crops}")

    # --- Сохранение результатов ---
    report.meta.total_time_ms = total_time
    saved = report.save(args.json)

    if args.csv:
        n_rows = save_csv(report, args.csv)
        print(f"CSV: {args.csv}  ({n_rows} строк)")

    print_class_summary(report)

    total_dets = report.meta.total_detections
    print(f"\nИтого: {report.meta.total_images} файлов, "
          f"{total_dets} детекций, "
          f"{total_time:.1f} ms  "
          f"({total_time / max(report.meta.total_images, 1):.1f} ms/img)")
    print(f"JSON: {saved}")

    # Интерактивный просмотр, если --show.
    if args.show and saved_results:
        try:
            browse_results(saved_results)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()