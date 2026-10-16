# main.py — детектор объектов и лиц на изображениях.
#
# Этот файл отвечает только за:
#   - разбор CLI-аргументов (и конфига);
#   - координацию работы: загрузка → инференс → сохранение.
#
# Логирование:
#   - прогресс и служебные сообщения → logger.info / logger.warning (stderr);
#   - итоговые результаты → print (stdout), видны даже при --quiet.

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np

from config_loader import load_config, merge_into_args
from cli_utils import (
    print_version, run_check, print_classes, print_presets,
)
from detector import (
    YoloDetector, Detection,
    WEIGHTS, CONFIG, NAMES, ONNX_MODEL,
    INPUT_SIZE, DEFAULT_CONF, DEFAULT_NMS, DEFAULT_CLASSES, CLASS_PRESETS,
)
from drawing import draw_detections, draw_faces, browse_results
from excel_report import save_excel
from face_detector import FaceDetector, FaceDetection, DEFAULT_MODEL
from io_utils import (
    save_crops, save_face_crops, save_csv,
    print_class_summary, collect_images,
)
from json_report import ReportBuilder, ReportMeta, ImageRecord, detection_to_dict
from logger import get_logger, setup_logging
from report_plots import save_report_plots


__version__ = "1.0.0"


# ---------- Утилиты ----------

def chunked(seq, n):
    """Разбивает последовательность на куски по n элементов."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------- CLI ----------

def build_parser():
    """Создаёт argparse-парсер со всеми аргументами."""
    p = argparse.ArgumentParser(description="YOLOv4-tiny object detector (images)")
    p.add_argument("--version", action="store_true",
                   help="показать версию и выйти")
    p.add_argument("--check", action="store_true",
                   help="проверить модели и окружение, выйти")
    p.add_argument("--list-classes", action="store_true",
                   help="показать все классы COCO, выйти")
    p.add_argument("--list-presets", action="store_true",
                   help="показать доступные пресеты, выйти")
    p.add_argument("inputs", nargs="*", default=["images/trucks.jpg"],
                   help="файлы, папки или glob-паттерны "
                        "(по умолчанию: images/trucks.jpg)")
    p.add_argument("--config", default=None,
                   help="путь к JSON-файлу с настройками")
    p.add_argument("--log-level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   default="INFO",
                   help="уровень логирования (по умолчанию INFO)")
    p.add_argument("--log-file", default=None,
                   help="писать лог в файл (например, output/run.log)")
    p.add_argument("--quiet", action="store_true",
                   help="то же, что --log-level WARNING")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--json", default="output/detections.json")
    p.add_argument("--csv", default="output/detections.csv",
                   help="путь к CSV (пустая строка — отключить)")
    p.add_argument("--xlsx", default="output/detections.xlsx",
                   help="путь к Excel (пустая строка — отключить)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--nms", type=float, default=DEFAULT_NMS)
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    p.add_argument("--preset", choices=list(CLASS_PRESETS.keys()))
    p.add_argument("--backend", choices=["opencv", "onnx"],
                   default="opencv",
                   help="бэкенд инференса (opencv DNN или onnxruntime)")
    p.add_argument("--tile", type=int, default=0, metavar="SIZE",
                   help="тайлинг для больших изображений: размер плитки "
                        "в пикселях (0 = выключено, рекомендуем 640)")
    p.add_argument("--tile-overlap", type=int, default=128,
                   help="перекрытие плиток при тайлинге (по умолчанию 128)")
    p.add_argument("--video", default=None, metavar="PATH",
                   help="путь к видеофайлу или RTSP-потоку")
    p.add_argument("--webcam", type=int, default=None, metavar="INDEX",
                   help="индекс веб-камеры (0, 1, ...) — включает видео-режим")
    p.add_argument("--save-video", default=None, metavar="PATH",
                   help="сохранить аннотированное видео")
    p.add_argument("--save-every", type=int, default=0,
                   help="сохранять каждый N-й кадр в output/frames")
    p.add_argument("--infer-every", type=int, default=1,
                   help="инференс раз в N кадров (ускорение для видео)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="остановиться после N кадров (0 = без лимита)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--crops-dir", default=None)
    p.add_argument("--faces", action="store_true")
    p.add_argument("--face-model", default=DEFAULT_MODEL)
    p.add_argument("--face-conf", type=float, default=0.7)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--plots-path", default="output/report.png")
    return p


def parse_args():
    """
    Разбор аргументов.
    Логирование настраивается ДО чтения конфига, чтобы все сообщения
    (включая предупреждения) шли через единый logger.
    Если указан --config, значения из JSON применяются как дефолты,
    но CLI-флаги остаются приоритетнее.
    """
    # 1. Предварительный парсинг: только --config и параметры логирования.
    # allow_abbrev=False — иначе --conf сокращается до --config.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--log-level", default="INFO")
    pre.add_argument("--log-file", default=None)
    pre.add_argument("--quiet", action="store_true")
    known, _ = pre.parse_known_args()

    # 2. Настраиваем логирование сразу
    level = "WARNING" if known.quiet else known.log_level
    setup_logging(level=level, log_file=known.log_file)
    logger = get_logger()

    # 3. Читаем конфиг — теперь сообщения идут через logger
    config = {}
    if known.config:
        try:
            config = load_config(known.config)
            logger.info(f"Конфиг: {known.config}  ({len(config)} ключей)")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(str(e))
            logger.warning("--config игнорируется")

    # 4. Полный парсинг
    parser = build_parser()
    defaults = parser.parse_args([])   # только дефолты — для сравнения
    args = parser.parse_args()

    # 5. --verbose поднимает уровень до DEBUG, если --log-level не задан явно
    if args.verbose and args.log_level == "INFO" and not args.quiet:
        logger.setLevel(logging.DEBUG)

    # 6. Мержим конфиг: CLI побеждает, конфиг заполняет «дырки»
    if config:
        args, applied = merge_into_args(args, config, defaults)
        if applied:
            logger.info(f"Из конфига применено: {', '.join(applied)}")

    return args


# ---------- Точка входа ----------

def main():
    args = parse_args()
    logger = get_logger()

    # --- Сервисные команды: выполняются и выходят ---
    if args.version:
        print_version(__version__)
        return

    if args.check:
        code = run_check({
            "YOLO weights": WEIGHTS,
            "YOLO config":  CONFIG,
            "YOLO names":   NAMES,
            "YuNet model":  DEFAULT_MODEL,
            "ONNX model":   ONNX_MODEL,
        })
        sys.exit(code)

    if args.list_classes:
        sys.exit(print_classes(NAMES))

    if args.list_presets:
        sys.exit(print_presets(CLASS_PRESETS))

    # --- Обычная обработка ---
    if not args.inputs:
        args.inputs = ["images/trucks.jpg"]

    if args.preset:
        classes = CLASS_PRESETS[args.preset]
    else:
        classes = tuple(args.classes)

    if args.backend == "onnx":
        from onnx_detector import OnnxDetector
        detector = OnnxDetector(
            model_path=ONNX_MODEL,
            names_path=NAMES,
            input_size=INPUT_SIZE,
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            classes_of_interest=classes,
            use_cuda=args.cuda,
        )
        logger.info(f"Бэкенд: onnx ({ONNX_MODEL})")
    else:
        detector = YoloDetector(
            WEIGHTS, CONFIG, NAMES,
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            classes_of_interest=classes,
            use_cuda=args.cuda,
        )
        logger.info(f"Бэкенд: opencv DNN ({WEIGHTS})")

    files = collect_images(args.inputs)
    if not files:
        logger.error("Не найдено ни одного изображения.")
        sys.exit(1)

    logger.info(f"Классов: {len(detector.classes)}  |  файлов: {len(files)}")
    logger.info(f"conf={args.conf}  nms={args.nms}  batch_size={args.batch_size}"
                + (f"  tile={args.tile}" if args.tile > 0 else ""))

    os.makedirs(args.output_dir, exist_ok=True)

    crops_dir = args.crops_dir or os.path.join(args.output_dir, "crops")
    crops_counter: dict = {}
    total_crops = 0
    if args.save_crops:
        os.makedirs(crops_dir, exist_ok=True)
        logger.info(f"Кропы: {crops_dir}")

    # --- Face-детектор (YuNet), если --faces ---
    face_detector = None
    if args.faces:
        try:
            face_detector = FaceDetector(
                model_path=args.face_model,
                score_threshold=args.face_conf,
            )
            logger.info(f"Face-детектор: {args.face_model}")
        except FileNotFoundError as e:
            logger.warning(str(e))
            logger.warning("--faces игнорируется")
    
        # --- Видео-режим: --video или --webcam ---
    video_source = None
    if args.webcam is not None:
        video_source = args.webcam
    elif args.video is not None:
        video_source = args.video

    if video_source is not None:
        from video_processor import process_video
        stats = process_video(
            detector=detector,
            source=video_source,
            face_detector=face_detector,
            output_video=args.save_video,
            show=not args.quiet and args.show,
            save_every=args.save_every,
            save_frames_dir=os.path.join(args.output_dir, "frames"),
            infer_every=args.infer_every,
            max_frames=args.max_frames,
        )
        print(f"\nВидео: кадров={stats.frames}  "
              f"детекций={stats.detections}  лиц={stats.faces}  "
              f"средний FPS={stats.fps_avg:.1f}")
        return

    # --- JSON-отчёт ---
    report = ReportBuilder(meta=ReportMeta(
        model=os.path.basename(os.path.dirname(WEIGHTS)),
        input_size=INPUT_SIZE,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        classes_of_interest=tuple(classes),
        batch_size=args.batch_size,
    ))

    total_time = 0.0
    saved_results: List[Tuple[str, np.ndarray]] = []

    # --- Загрузка и группировка по размеру ---
    by_size: dict = defaultdict(list)
    for f in files:
        img = cv2.imread(f)
        if img is None:
            logger.warning(f"не удалось прочитать: {f}")
            continue
        by_size[img.shape[:2]].append((f, img))

    if not by_size:
        logger.error("Не удалось прочитать ни одного изображения.")
        sys.exit(1)

    sizes_info = ", ".join(f"{h}x{w}: {len(v)}"
                           for (h, w), v in by_size.items())
    logger.info(f"Групп по размеру: {len(by_size)} ({sizes_info})")

    # --- Основной цикл ---
    batch_idx = 0
    for (grp_h, grp_w), items in by_size.items():
        for batch_files in chunked(items, args.batch_size):
            batch_idx += 1
            valid_files = [f for f, _ in batch_files]
            images = [img for _, img in batch_files]

            t0 = time.perf_counter()
            if args.tile > 0:
                batch_dets = [
                    detector.detect_tiled(img, args.tile, args.tile_overlap)
                    for img in images
                ]
            else:
                batch_dets = detector.detect_batch(images)
            dt_batch = (time.perf_counter() - t0) * 1000
            total_time += dt_batch
            per_img_ms = dt_batch / len(images)

            logger.info(f"[batch {batch_idx}] group {grp_h}x{grp_w}  "
                        f"{len(images)} img  "
                        f"{dt_batch:.1f} ms total  ({per_img_ms:.1f} ms/img)")

            for f, img, dets in zip(valid_files, images, batch_dets):
                h, w = img.shape[:2]
                name = os.path.basename(f)
                logger.info(f"  {name:<30} найдено: {len(dets)}")

                if args.verbose:
                    for d in sorted(dets, key=lambda x: -x.score):
                        logger.debug(f"      {d.class_name:<12} "
                                     f"score={d.score:.3f} box={d.box}")

                # YOLO-боксы
                result = draw_detections(img, dets)

                # Детекция лиц поверх YOLO
                faces: List[FaceDetection] = []
                if face_detector is not None:
                    faces = face_detector.detect(img)
                    if args.verbose and faces:
                        for fd in faces:
                            logger.debug(f"      {'face':<12} "
                                         f"score={fd.score:.3f} box={fd.box}")
                    result = draw_faces(result, faces)

                    count_text = f"Faces: {len(faces)}"
                    (tw2, _), _ = cv2.getTextSize(
                        count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(result, count_text,
                                (result.shape[1] - tw2 - 10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2, cv2.LINE_AA)

                out_path = os.path.join(args.output_dir, name)
                cv2.imwrite(out_path, result)
                saved_results.append((name, result))

                # Кропы: YOLO-объекты + лица
                if args.save_crops:
                    total_crops += save_crops(img, dets, name,
                                              crops_dir, crops_counter)
                    if faces:
                        total_crops += save_face_crops(
                            img, faces, name, crops_dir, crops_counter)

                # JSON: YOLO + лица (класс "face")
                all_dets = [detection_to_dict(d) for d in dets]
                for fd in faces:
                    all_dets.append({
                        "class_id": -1,
                        "class_name": "face",
                        "score": round(float(fd.score), 4),
                        "box": {"x": int(fd.x1), "y": int(fd.y1),
                                "w": int(fd.box[2]), "h": int(fd.box[3])},
                        "xyxy": [int(fd.x1), int(fd.y1),
                                 int(fd.x2), int(fd.y2)],
                    })

                report.add_image(ImageRecord(
                    path=f,
                    output=out_path,
                    width=w,
                    height=h,
                    inference_ms=per_img_ms,
                    detections=all_dets,
                ))

    if args.save_crops:
        print(f"Кропов сохранено: {total_crops}")

    # --- Сохранение результатов ---
    report.meta.total_time_ms = total_time
    saved = report.save(args.json)

    if args.csv:
        n_rows = save_csv(report, args.csv)
        print(f"CSV: {args.csv}  ({n_rows} строк)")

    if args.xlsx:
        xlsx_path = save_excel(report, args.xlsx)
        if xlsx_path:
            print(f"XLSX: {xlsx_path}")

    print_class_summary(report)

    if args.plots:
        plot_path = save_report_plots(report, args.plots_path)
        if plot_path:
            print(f"Графики: {plot_path}")

    total_dets = report.meta.total_detections
    print(f"\nИтого: {report.meta.total_images} файлов, "
          f"{total_dets} детекций, "
          f"{total_time:.1f} ms  "
          f"({total_time / max(report.meta.total_images, 1):.1f} ms/img)")
    print(f"JSON: {saved}")

    if args.show and saved_results:
        try:
            browse_results(saved_results)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()