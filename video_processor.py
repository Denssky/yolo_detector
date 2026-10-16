"""
Обработка видео и веб-камеры.
Читает кадры через cv2.VideoCapture, детектит, рисует боксы,
опционально пишет результат и показывает в окне.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from detector import Detection
from drawing import draw_detections, draw_faces, FACE_COLOR
from face_detector import FaceDetection
from logger import get_logger


@dataclass
class VideoStats:
    """Статистика обработки видео."""
    frames: int = 0
    detections: int = 0
    faces: int = 0
    total_infer_ms: float = 0.0
    total_frame_ms: float = 0.0
    fps_avg: float = 0.0

    def add(self, infer_ms: float, frame_ms: float,
            n_dets: int, n_faces: int):
        self.frames += 1
        self.detections += n_dets
        self.faces += n_faces
        self.total_infer_ms += infer_ms
        self.total_frame_ms += frame_ms
        if self.total_frame_ms > 0:
            self.fps_avg = 1000.0 * self.frames / self.total_frame_ms


def open_source(source) -> cv2.VideoCapture:
    """
    source: путь к файлу (str) или индекс камеры (int).
    """
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        kind = f"камера {source}"
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Файл не найден: {source}")
        cap = cv2.VideoCapture(source)
        kind = f"файл {source}"

    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть источник: {kind}. "
            f"Возможные причины: файла нет, нет кодека (проверьте FFMPEG "
            f"в cv2.getBuildInformation()), камера занята другим приложением."
        )
    return cap


def process_video(
    detector,
    source,
    face_detector=None,
    output_video: Optional[str] = None,
    show: bool = True,
    save_every: int = 0,
    save_frames_dir: str = "output/frames",
    infer_every: int = 1,
    max_frames: int = 0,
    window_name: str = "YOLO video",
    print_every: int = 30,
) -> VideoStats:
    """
    Основной цикл видео.

    source: путь к файлу или индекс камеры (int).
    infer_every: детектить раз в N кадров (1 = каждый кадр).
    save_every: сохранять каждый N-й кадр в save_frames_dir (0 = не сохранять).
    max_frames: остановиться после N кадров (0 = без лимита).
    """
    logger = get_logger()
    cap = open_source(source)
    stats = VideoStats()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    logger.info(f"Источник: {source}  |  {width}x{height}  |  "
                f"FPS источника: {src_fps:.1f}")

    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = src_fps if src_fps and src_fps > 1 else 25.0
        os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
        writer = cv2.VideoWriter(output_video, fourcc, out_fps,
                                 (width, height))
        if not writer.isOpened():
            logger.warning(f"не удалось открыть VideoWriter: {output_video}")
            writer = None
        else:
            logger.info(f"Запись видео: {output_video}  @ {out_fps:.1f} FPS")

    if save_every > 0:
        os.makedirs(save_frames_dir, exist_ok=True)
        logger.info(f"Кадры будут сохраняться в: {save_frames_dir}")

    frame_idx = 0
    last_dets = []
    last_faces = []
    last_infer_ms = 0.0
    t_start = time.perf_counter()

    try:
        while True:
            t_frame = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                logger.info("Конец потока.")
                break
            frame_idx += 1

            # --- Детекция раз в infer_every кадров ---
            did_infer = (infer_every <= 1
                         or frame_idx % infer_every == 0
                         or not last_dets)
            if did_infer:
                t0 = time.perf_counter()
                last_dets = detector.detect(frame)
                if face_detector is not None:
                    last_faces = face_detector.detect(frame)
                last_infer_ms = (time.perf_counter() - t0) * 1000

            dets = last_dets
            faces = last_faces

            # --- Отрисовка ---
            annotated = frame.copy()
            annotated = draw_detections(annotated, dets)
            if face_detector is not None:
                annotated = draw_faces(annotated, faces)
                if faces:
                    txt = f"Faces: {len(faces)}"
                    (tw, _), _ = cv2.getTextSize(
                        txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(annotated, txt,
                                (annotated.shape[1] - tw - 10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                FACE_COLOR, 2, cv2.LINE_AA)

            frame_ms = (time.perf_counter() - t_frame) * 1000
            stats.add(last_infer_ms, frame_ms, len(dets), len(faces))

            # --- HUD ---
            infer_mark = "INF" if did_infer else "···"
            hud = (f"frame={frame_idx}  {infer_mark}  "
                   f"det={len(dets)}  "
                   f"infer={last_infer_ms:.1f}ms  "
                   f"fps={stats.fps_avg:.1f}")
            cv2.putText(annotated, hud, (10, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, hud, (10, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(annotated)

            if save_every > 0 and frame_idx % save_every == 0:
                out_path = os.path.join(save_frames_dir,
                                        f"frame_{frame_idx:07d}.jpg")
                cv2.imwrite(out_path, annotated)

            if print_every > 0 and frame_idx % print_every == 0:
                logger.info(f"  frame={frame_idx:>6}  {infer_mark}  "
                            f"det={len(dets):>3}  "
                            f"infer={last_infer_ms:6.1f}ms  "
                            f"fps={stats.fps_avg:5.1f}")

            if show:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info(f"Остановлено пользователем на кадре {frame_idx}.")
                    break

            if max_frames and frame_idx >= max_frames:
                logger.info(f"Достигнут лимит кадров: {max_frames}.")
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info(f"Видео сохранено: {output_video}")
        if show:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - t_start
    avg_infer = stats.total_infer_ms / max(stats.frames, 1)
    logger.info(f"Итого: кадров={stats.frames}  детекций={stats.detections}  "
                f"лиц={stats.faces}  средний FPS={stats.fps_avg:.1f}  "
                f"средний инференс={avg_infer:.1f}ms  "
                f"wall={wall:.1f}s")
    return stats