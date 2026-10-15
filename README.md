# YOLO Object Detector

Детекция объектов и лиц на изображениях с помощью YOLOv4-tiny, YuNet и OpenCV DNN.
По умолчанию ищет грузовики, но умеет работать с любыми классами COCO
(80 классов) и опционально — с лицами.

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install opencv-python numpy matplotlib openpyxl

# 2. Скачать модель YOLOv4-tiny в папку yolo_tiny/
#    (ссылки — в разделе «Установка»)

# 3. Запустить на тестовом изображении
python main.py images/trucks.jpg
```

Результат — в `output/`: аннотированные картинки, `detections.json`, `detections.csv`,
`detections.xlsx`.

---

## Установка

### 1. Зависимости

```bash
pip install opencv-python numpy matplotlib openpyxl
```

Python 3.8+ обязателен. Для CUDA-инференса (опционально) нужна сборка OpenCV с CUDA.

### 2. Модель YOLOv4-tiny

Скачайте три файла в папку `yolo_tiny/`:

| Файл | Ссылка |
|---|---|
| `yolov4-tiny.weights` | https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights |
| `yolov4-tiny.cfg` | https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg |
| `coco.names` | https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names |

Одной командой PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path yolo_tiny | Out-Null
Invoke-WebRequest "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights" -OutFile "yolo_tiny\yolov4-tiny.weights"
Invoke-WebRequest "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg" -OutFile "yolo_tiny\yolov4-tiny.cfg"
Invoke-WebRequest "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names" -OutFile "yolo_tiny\coco.names"
```

### 3. Модель YuNet (только для `--faces`)

```powershell
New-Item -ItemType Directory -Force -Path models | Out-Null
Invoke-WebRequest "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models\face_detection_yunet_2023mar.onnx"
```

---

## Использование

### Базовый запуск

```bash
python main.py images/trucks.jpg       # один файл
python main.py images/                 # папка
python main.py "images/**/*.jpg"       # glob
```

### Пресеты классов

```bash
python main.py images/ --preset traffic    # люди, машины, светофоры
python main.py images/ --preset office     # ноутбуки, телефоны, книги
python main.py images/ --preset food       # еда
python main.py images/ --preset all        # все 80 классов COCO
```

### Свой список классов

```bash
python main.py images/ --classes person "cell phone" laptop
```

### Кропы объектов

```bash
python main.py images/ --preset traffic --save-crops
# output/crops/trucks_truck_001.jpg, ...
# при --faces: lena_face_001.jpg
```

### Детекция лиц

```bash
python main.py images/ --faces
python main.py images/ --faces --face-conf 0.9
```

### Графики

```bash
python main.py images/ --preset traffic --plots
# output/report.png — 4 графика в сетке 2×2
```

### Просмотр результатов

```bash
python main.py images/ --show
# A / [ — назад
# D / ] / Space — вперёд
# Q / Esc — выход
```

### Конфиг-файл

Вместо длинных команд — JSON со всеми настройками.

```powershell
Copy-Item settings.example.json settings.json
# отредактируйте settings.json под себя
python main.py images\ --config settings.json
```

Пример `settings.json`:

```json
{
  "preset": "traffic",
  "conf": 0.2,
  "batch_size": 8,
  "save_crops": true,
  "faces": true,
  "face_conf": 0.9,
  "plots": true,
  "verbose": false
}
```

**CLI-флаги приоритетнее конфига.** Если передали `--conf 0.5`, оно перебьёт
значение из файла:

```powershell
python main.py images\ --config settings.json --conf 0.5
```

Поддерживаемые ключи — в `config_loader.py` (`CONFIG_KEY_MAP`). Неизвестные
ключи игнорируются с предупреждением.

### Тонкая настройка

```bash
python main.py images/ --conf 0.2                # больше детекций
python main.py images/ --batch-size 16           # размер батча
python main.py images/ --cuda                    # GPU
python main.py images/ --verbose                 # печатать каждую детекцию
python main.py images/ --csv ""                  # отключить CSV
python main.py images/ --xlsx ""                 # отключить Excel
```

---

## Все флаги

| Флаг | Описание | По умолчанию |
|---|---|---|
| `inputs` | Файлы, папки или glob-паттерны | `images/trucks.jpg` |
| `--config` | Путь к JSON-файлу с настройками | — |
| `--preset` | Готовый набор классов | — |
| `--classes` | Произвольный список классов COCO | `truck` |
| `--conf` | Порог уверенности | `0.4` |
| `--nms` | Порог NMS | `0.4` |
| `--batch-size` | Размер батча | `8` |
| `--output-dir` | Куда сохранять результаты | `output` |
| `--json` | Путь к JSON | `output/detections.json` |
| `--csv` | Путь к CSV (пустая строка — отключить) | `output/detections.csv` |
| `--xlsx` | Путь к Excel (пустая строка — отключить) | `output/detections.xlsx` |
| `--save-crops` | Сохранять кропы объектов и лиц | выкл. |
| `--crops-dir` | Папка для кропов | `<output-dir>/crops` |
| `--faces` | Детектировать лица (YuNet) | выкл. |
| `--face-model` | Путь к ONNX-модели YuNet | `models/face_detection_yunet_2023mar.onnx` |
| `--face-conf` | Порог уверенности для лиц | `0.7` |
| `--plots` | Построить графики | выкл. |
| `--plots-path` | Путь к PNG с графиками | `output/report.png` |
| `--show` | Открыть окно с результатом | выкл. |
| `--verbose` | Печатать каждую детекцию | выкл. |
| `--cuda` | Использовать CUDA | выкл. |

---

## Что получается на выходе

```
output/
├── trucks.jpg              # аннотированное изображение
├── detections.json         # полный отчёт: метаданные + все детекции
├── detections.csv          # плоская таблица для pandas
├── detections.xlsx         # Excel: листы «Detections» и «Summary»
├── report.png              # графики (при --plots)
└── crops/                  # кропы (при --save-crops)
    ├── trucks_truck_001.jpg
    └── lena_face_001.jpg
```

**Формат JSON:**

```json
{
  "meta": {
    "model": "yolo_tiny",
    "conf_threshold": 0.4,
    "total_images": 3,
    "total_detections": 7,
    "total_time_ms": 512.3
  },
  "images": [
    {
      "path": "images/trucks.jpg",
      "output": "output/trucks.jpg",
      "width": 1024,
      "height": 640,
      "detections": [
        {
          "class_name": "truck",
          "class_id": 7,
          "score": 0.9721,
          "box": {"x": 661, "y": 371, "w": 139, "h": 153},
          "xyxy": [661, 371, 800, 524]
        },
        {
          "class_name": "face",
          "class_id": -1,
          "score": 0.5432,
          "box": {"x": 207, "y": 183, "w": 146, "h": 205},
          "xyxy": [207, 183, 353, 388]
        }
      ]
    }
  ]
}
```

**CSV:** `file, class_name, class_id, score, x, y, w, h`

**Excel:** два листа — «Detections» (все строки) и «Summary» (сводка
по классам: количество, средний/min/max score).

---

## Структура проекта

```
yolo_detector/
├── main.py               # точка входа, CLI, обработка изображений
├── config_loader.py      # чтение settings.json
├── json_report.py        # сборка JSON-отчёта
├── face_detector.py      # YuNet для детекции лиц
├── report_plots.py       # графики matplotlib
├── excel_report.py       # Excel-экспорт через openpyxl
├── settings.example.json # шаблон конфига
├── yolo_tiny/            # модель YOLO
├── models/               # ONNX-модель YuNet
├── images/               # входные изображения
└── output/               # результаты
```

---

## Классы COCO (80)

Полный список — в `yolo_tiny/coco.names`. Основные:

- **Люди:** `person`
- **Транспорт:** `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`
- **Улица:** `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`
- **Животные:** `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`
- **Аксессуары:** `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`
- **Спорт:** `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`
- **Кухня:** `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`
- **Еда:** `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`
- **Мебель:** `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`
- **Электроника:** `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`
- **Бытовая техника:** `microwave`, `oven`, `toaster`, `sink`, `refrigerator`
- **Разное:** `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`

**Плюс псевдо-класс `face`** (от YuNet, при `--faces`).

---

## Требования

- Python 3.8+
- OpenCV 4.5.4+
- numpy, matplotlib, openpyxl

## Лицензия

MIT