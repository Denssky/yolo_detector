# YOLO Object Detector

Детекция объектов и лиц на изображениях с помощью YOLOv4-tiny, YuNet и OpenCV DNN.
Два бэкенда инференса: OpenCV DNN и ONNX Runtime (в ~1.8x быстрее).
Поддержка тайлинга для больших изображений, JSON/CSV/Excel-отчёты,
веб-просмотрщик, 34 теста.

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install opencv-python numpy matplotlib openpyxl flask

# 2. Скачать модель YOLOv4-tiny в папку yolo_tiny/
#    (ссылки — в разделе «Установка»)

# 3. Запустить на тестовом изображении
python main.py images/trucks.jpg
```

Результат — в `output/`: аннотированные картинки, `detections.json`,
`detections.csv`, `detections.xlsx`.

---

## Установка

### 1. Зависимости

```bash
pip install opencv-python numpy matplotlib openpyxl flask
# опционально — для --backend onnx
pip install onnxruntime
```

Python 3.8+ обязателен. Для CUDA-инференса (опционально) нужна сборка
OpenCV с поддержкой CUDA.

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

### 4. Модель ONNX (только для `--backend onnx`)

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/opencv/opencv_contribution/resolve/main/yolov4/yolov4-tiny.onnx" -OutFile "yolo_tiny\yolov4-tiny.onnx"
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
python main.py images/ --classes truck car bus
```

### Ускорение через ONNX Runtime

```bash
python main.py images/ --backend onnx
```

ONNX Runtime даёт **~1.8x ускорение** на CPU по сравнению с OpenCV DNN.
Результаты могут слегка отличаться (float32-расхождения, разные
реализации постобработки).

### Тайлинг для больших изображений

```bash
python main.py images/ --tile 640
python main.py images/ --tile 320 --tile-overlap 64
```

Для изображений больше 1000 px по стороне помогает найти **мелкие
объекты** — людей вдалеке, светофоры, знаки. Тайлинг режет изображение
на плитки `--tile` px с перекрытием `--tile-overlap` и применяет
глобальный NMS.

**Компромисс:** в 3–5 раз больше детекций, но и в 5–20 раз дольше.
`--tile 640` — оптимальный баланс.

### Кропы объектов и лиц

```bash
python main.py images/ --preset traffic --save-crops
# output/crops/trucks_truck_001.jpg
# при --faces: lena_face_001.jpg
```

### Детекция лиц

```bash
python main.py images/ --faces
python main.py images/ --faces --face-conf 0.9
```

Лица рисуются **жёлтым**, ключевые точки (глаза, нос, уголки рта) —
**красным**. В сводке появляется `face`.

### Графики результатов

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

### Веб-просмотрщик

```bash
python web_viewer.py
# Открыть http://127.0.0.1:5000
```

Галерея обработанных изображений, сводка по классам, JSON-API
(`/api/data`). HTML и CSS вынесены в `templates/` и `static/`.

### Видео и веб-камера

```bash
# Видеофайл
python main.py --video samples/vtest.avi --show
python main.py --video samples/vtest.avi --save-video output/result.mp4

# Веб-камера
python main.py --webcam 0 --show

# Ускорение: детектить раз в N кадров
python main.py --video samples/vtest.avi --infer-every 3 --save-video output/fast.mp4

# Ограничить обработку
python main.py --video samples/vtest.avi --max-frames 100 --quiet
```

**Управление в окне:** `Q` / `Esc` — выход.

**Флаги для видео:**

| Флаг | Описание | По умолчанию |
|---|---|---|
| `--video PATH` | Путь к видеофайлу или RTSP-потоку | — |
| `--webcam INDEX` | Индекс веб-камеры (0, 1, ...) | — |
| `--save-video PATH` | Сохранить аннотированное видео | — |
| `--save-every N` | Сохранять каждый N-й кадр в `output/frames` | `0` |
| `--infer-every N` | Инференс раз в N кадров (ускорение) | `1` |
| `--max-frames N` | Остановиться после N кадров | `0` |

**HUD в окне:** `frame=NNN`, `INF`/`···` (был ли инференс на кадре),
`det=N`, `infer=Xms`, `fps=Y`.

### Конфиг-файл

Вместо длинных команд — JSON:

```powershell
Copy-Item settings.example.json settings.json
# отредактируйте settings.json
python main.py images\ --config settings.json
```

Пример `settings.json`:

```json
{
  "preset": "traffic",
  "conf": 0.2,
  "backend": "onnx",
  "tile": 640,
  "faces": true,
  "plots": true
}
```

**CLI-флаги приоритетнее конфига.** Если передали `--conf 0.5`,
оно перебьёт значение из файла.

Поддерживаемые ключи — в `config_loader.py` (`CONFIG_KEY_MAP`).

### Логирование

```bash
python main.py images/ --log-level DEBUG
python main.py images/ --log-file output/run.log
python main.py images/ --quiet                 # то же, что WARNING
```

Прогресс идёт в **stderr**, итоговые результаты — в **stdout**.
При `--quiet` видны только предупреждения и ошибки.
`--log-file` пишет всё с датами и временем.

### Сервисные команды

```bash
python main.py --version           # версия и версии библиотек
python main.py --check             # проверить модели и окружение
python main.py --list-classes      # все 80 классов COCO
python main.py --list-presets      # доступные пресеты
```

`--check` полезен, если что-то не работает: сразу видно, какие модели
не найдены и есть ли FaceDetectorYN.

### Тонкая настройка

```bash
python main.py images/ --conf 0.2                # больше детекций
python main.py images/ --batch-size 16           # размер батча
python main.py images/ --cuda                    # GPU, если доступна
python main.py images/ --verbose                 # детализация (DEBUG)
python main.py images/ --csv ""                  # отключить CSV
python main.py images/ --xlsx ""                 # отключить Excel
```

---

## Все флаги

| Флаг | Описание | По умолчанию |
|---|---|---|
| `inputs` | Файлы, папки или glob-паттерны | `images/trucks.jpg` |
| `--config` | Путь к JSON-файлу с настройками | — |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |
| `--log-file` | Писать лог в файл | — |
| `--quiet` | То же, что `--log-level WARNING` | выкл. |
| `--preset` | Готовый набор классов | — |
| `--classes` | Произвольный список классов COCO | `truck` |
| `--conf` | Порог уверенности | `0.4` |
| `--nms` | Порог NMS | `0.4` |
| `--backend` | `opencv` или `onnx` | `opencv` |
| `--tile` | Размер плитки для тайлинга (0 = выкл.) | `0` |
| `--tile-overlap` | Перекрытие плиток | `128` |
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
| `--verbose` | Печатать каждую детекцию (DEBUG) | выкл. |
| `--cuda` | Использовать CUDA | выкл. |
| `--version` | Показать версию и выйти | — |
| `--check` | Проверить окружение и выйти | — |
| `--list-classes` | Показать 80 классов COCO | — |
| `--list-presets` | Показать пресеты | — |

---

## Что получается на выходе

```
output/
├── trucks.jpg              # аннотированное изображение
├── detections.json         # полный отчёт
├── detections.csv          # плоская таблица для pandas
├── detections.xlsx         # Excel: Detections + Summary
├── report.png              # графики (при --plots)
└── crops/                  # кропы (при --save-crops)
    ├── trucks_truck_001.jpg
    └── lena_face_001.jpg
```

**JSON** содержит `meta` (модель, пороги, время) и `images` — список
с `path`, `output`, `width`, `height`, `detections`. Каждая детекция:
`class_id`, `class_name`, `score`, `box {x, y, w, h}`, `xyxy`.

**CSV:** `file, class_name, class_id, score, x, y, w, h`

**Excel:** два листа — «Detections» (все строки) и «Summary»
(сводка по классам).

---

## Структура проекта

```
yolo_detector/
├── main.py               # CLI + координация
├── detector.py           # YoloDetector (OpenCV DNN)
├── onnx_detector.py      # OnnxDetector (ONNX Runtime)
├── tiling.py             # тайлинг больших изображений
├── drawing.py            # отрисовка, просмотр
├── io_utils.py           # кропы, CSV, сбор файлов
├── face_detector.py      # YuNet
├── json_report.py        # JSON-отчёт
├── excel_report.py       # Excel через openpyxl
├── report_plots.py       # графики matplotlib
├── config_loader.py      # чтение settings.json
├── cli_utils.py          # --version, --check, --list-*
├── logger.py             # логирование
├── web_viewer.py         # Flask-просмотрщик
├── tests/                # pytest: 34 теста
├── templates/, static/   # HTML/CSS для Flask
├── settings.example.json # шаблон конфига
├── yolo_tiny/            # модель YOLO
├── models/               # YuNet ONNX
├── images/, output/
└── README.md
```

**Зависимости модулей** (без циклов):

```
detector.py        → (ничего)
tiling.py          → detector
drawing.py         → detector, face_detector
io_utils.py        → detector, face_detector, json_report, logger
main.py            → всё выше + config_loader, excel_report, report_plots, cli_utils, logger
onnx_detector.py   → detector, tiling
web_viewer.py      → только output/detections.json
```

---

## Тесты

```bash
pip install pytest
pytest
```

34 теста покрывают ключевые модули: сбор файлов, чтение конфига,
постобработку YOLO, сохранение кропов, CLI-утилиты, логирование.

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
- numpy, matplotlib, openpyxl, flask
- onnxruntime (для `--backend onnx`)
- pytest (для тестов)

## Лицензия

MIT