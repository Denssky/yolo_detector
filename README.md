# YOLO Truck Detector

Детекция объектов (грузовики, транспорт, произвольные классы COCO) на изображениях
с помощью YOLOv4-tiny и OpenCV DNN.

## Возможности

- Детекция 80 классов COCO через YOLOv4-tiny
- Пакетная обработка: файлы, папки, glob-паттерны
- Пресеты классов (`--preset traffic|office|food|all`)
- Фильтр по произвольным классам (`--classes ...`)
- Сохранение аннотированных изображений и сводного JSON-отчёта
- Батч-инференс, опциональный CUDA

## Установка

```bash
pip install opencv-python numpy
```

Скачайте модель YOLOv4-tiny и положите в `yolo_tiny/`:

- `yolov4-tiny.weights`
- `yolov4-tiny.cfg`
- `coco.names`

## Использование

```bash
# По умолчанию — грузовики в images/trucks.jpg
python main.py

# Пресет "трафик"
python main.py images/ --preset traffic

# Свой список классов
python main.py images/ --classes person car bottle

# С окном
python main.py images/ --show
```

## Аргументы

| Флаг | Описание |
|---|---|
| `inputs` | Файлы, папки или glob-паттерны |
| `--preset` | Готовый набор классов |
| `--classes` | Произвольный список классов COCO |
| `--conf` | Порог уверенности (по умолчанию 0.4) |
| `--nms` | Порог NMS (по умолчанию 0.4) |
| `--batch-size` | Размер батча (по умолчанию 8) |
| `--output-dir` | Куда сохранять результаты |
| `--json` | Путь к сводному JSON |
| `--show` | Открыть окно с последним результатом |
| `--verbose` | Печатать все детекции |
| `--cuda` | Использовать CUDA, если доступна |

## Модель

Скачайте YOLOv4-tiny и положите в `yolo_tiny/`:
- https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
- https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
- https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
## Лицензия
github.com/AlexeyAB/darknet
MIT