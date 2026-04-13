from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


def _load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Failed to read image: {path}")

    return frame


def _build_hog_detector() -> cv2.HOGDescriptor:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _should_keep_detection(
    width: int,
    height: int,
    confidence: float,
    *,
    min_width: int,
    min_height: int,
    min_confidence: float,
    min_area: int,
    max_aspect_ratio: float,
) -> bool:
    if width < min_width:
        return False
    if height < min_height:
        return False
    if width * height < min_area:
        return False
    if confidence < min_confidence:
        return False

    aspect_ratio = height / max(width, 1)
    if aspect_ratio > max_aspect_ratio:
        return False

    return True


def _draw_detections(
    image: np.ndarray,
    boxes: list[list[int]],
    *,
    title: str,
) -> np.ndarray:
    output = image.copy()

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        output,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def _build_debug_image_path(processed_image_path: str | Path) -> Path:
    processed_path = Path(processed_image_path)
    stem = processed_path.stem
    suffix = processed_path.suffix or ".jpg"
    return processed_path.parent / f"{stem}_debug{suffix}"


def save_debug_image(
    *,
    processed_image_path: str | Path,
    boxes: list[list[int]],
    title: str,
) -> str:
    frame = _load_image(processed_image_path)
    annotated = _draw_detections(frame, boxes, title=title)

    output_path = _build_debug_image_path(processed_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), annotated)
    if not ok:
        raise RuntimeError(f"Failed to save debug image: {output_path}")

    logger.info("Saved debug image: path=%s", output_path)
    return str(output_path)


def detect_people_local(
    image_path: str | Path,
    *,
    min_width: int = 48,
    min_height: int = 96,
    min_confidence: float = 0.2,
    min_area: int = 48 * 96,
    max_aspect_ratio: float = 5.0,
    unknown_confidence_threshold: float = 0.3,
    save_debug: bool = False,
) -> dict[str, Any]:
    frame = _load_image(image_path)
    height, width = frame.shape[:2]

    mean_brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    image_too_dark = mean_brightness < 20.0

    hog = _build_hog_detector()
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    raw_boxes: list[list[int]] = []
    raw_confidence_scores: list[float] = []

    for (x, y, w, h), weight in zip(rects, weights):
        raw_boxes.append([int(x), int(y), int(x + w), int(y + h)])
        raw_confidence_scores.append(float(weight))

    filtered_boxes: list[list[int]] = []
    filtered_confidence_scores: list[float] = []

    for (x, y, w, h), weight in zip(rects, weights):
        confidence = float(weight)
        if _should_keep_detection(
            int(w),
            int(h),
            confidence,
            min_width=min_width,
            min_height=min_height,
            min_confidence=min_confidence,
            min_area=min_area,
            max_aspect_ratio=max_aspect_ratio,
        ):
            filtered_boxes.append([int(x), int(y), int(x + w), int(y + h)])
            filtered_confidence_scores.append(confidence)

    filtered_count = len(filtered_boxes)
    max_confidence = max(filtered_confidence_scores) if filtered_confidence_scores else 0.0

    if image_too_dark:
        scene_state = "unknown"
        person_count_estimate = None
        confidence = 0.0
    elif filtered_count == 0:
        scene_state = "empty"
        person_count_estimate = 0
        confidence = 0.8
    elif max_confidence < unknown_confidence_threshold:
        scene_state = "unknown"
        person_count_estimate = None
        confidence = _clamp_confidence(max_confidence)
    elif filtered_count == 1:
        scene_state = "occupied"
        person_count_estimate = 1
        confidence = _clamp_confidence(max_confidence)
    else:
        scene_state = "occupied"
        person_count_estimate = "2+"
        confidence = _clamp_confidence(max_confidence)

    debug_image_path: str | None = None
    if save_debug:
        title = f"scene={scene_state} count={person_count_estimate or 'unknown'}"
        debug_image_path = save_debug_image(
            processed_image_path=image_path,
            boxes=filtered_boxes,
            title=title,
        )

    result = {
        "scene_state": scene_state,
        "person_count_estimate": person_count_estimate,
        "confidence": confidence,
        "signals": {
            "detector_type": "opencv_hog",
            "image_width": width,
            "image_height": height,
            "mean_brightness": mean_brightness,
            "image_too_dark": image_too_dark,
            "raw_detection_count": len(raw_boxes),
            "filtered_detection_count": filtered_count,
            "raw_boxes": raw_boxes,
            "filtered_boxes": filtered_boxes,
            "raw_confidence_scores": raw_confidence_scores,
            "filtered_confidence_scores": filtered_confidence_scores,
            "thresholds": {
                "min_width": min_width,
                "min_height": min_height,
                "min_confidence": min_confidence,
                "min_area": min_area,
                "max_aspect_ratio": max_aspect_ratio,
                "unknown_confidence_threshold": unknown_confidence_threshold,
            },
            "debug_image_path": debug_image_path,
        },
    }

    logger.info(
        "Detection completed: image=%s scene_state=%s person_count_estimate=%s raw_detection_count=%s filtered_detection_count=%s",
        image_path,
        scene_state,
        person_count_estimate,
        len(raw_boxes),
        filtered_count,
    )
    return result