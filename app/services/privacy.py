from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from app.logger import get_logger

logger = get_logger(__name__)

Box = tuple[int, int, int, int]


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def crop_roi(frame: np.ndarray, roi: Box | None = None) -> tuple[np.ndarray, dict]:
    if roi is None:
        return frame.copy(), {
            "roi_applied": False,
            "roi_box": None,
        }

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(roi, width, height)

    if x1 == x2 or y1 == y2:
        raise ValueError(f"Invalid ROI after clamping: {(x1, y1, x2, y2)}")

    cropped = frame[y1:y2, x1:x2].copy()

    logger.info("Applied ROI crop: roi=%s result_shape=%s", (x1, y1, x2, y2), cropped.shape)
    return cropped, {
        "roi_applied": True,
        "roi_box": [x1, y1, x2, y2],
    }


def apply_mask_regions(
    frame: np.ndarray,
    mask_regions: Iterable[Box] | None = None,
) -> tuple[np.ndarray, dict]:
    if not mask_regions:
        return frame.copy(), {
            "mask_applied": False,
            "mask_regions": [],
        }

    output = frame.copy()
    height, width = output.shape[:2]
    applied_regions: list[list[int]] = []

    for region in mask_regions:
        x1, y1, x2, y2 = clamp_box(region, width, height)
        if x1 == x2 or y1 == y2:
            continue

        output[y1:y2, x1:x2] = 0
        applied_regions.append([x1, y1, x2, y2])

    logger.info("Applied mask regions: count=%s", len(applied_regions))
    return output, {
        "mask_applied": len(applied_regions) > 0,
        "mask_regions": applied_regions,
    }


def apply_blur_regions(
    frame: np.ndarray,
    blur_regions: Iterable[Box] | None = None,
    kernel_size: tuple[int, int] = (31, 31),
) -> tuple[np.ndarray, dict]:
    if not blur_regions:
        return frame.copy(), {
            "blur_applied": False,
            "blur_regions": [],
        }

    output = frame.copy()
    height, width = output.shape[:2]
    applied_regions: list[list[int]] = []

    kx, ky = kernel_size
    if kx % 2 == 0 or ky % 2 == 0:
        raise ValueError("Blur kernel_size must use odd numbers, e.g. (31, 31)")

    for region in blur_regions:
        x1, y1, x2, y2 = clamp_box(region, width, height)
        if x1 == x2 or y1 == y2:
            continue

        roi = output[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, kernel_size, 0)
        output[y1:y2, x1:x2] = blurred
        applied_regions.append([x1, y1, x2, y2])

    logger.info("Applied blur regions: count=%s", len(applied_regions))
    return output, {
        "blur_applied": len(applied_regions) > 0,
        "blur_regions": applied_regions,
        "blur_kernel_size": [kx, ky],
    }


def apply_privacy_pipeline(
    frame: np.ndarray,
    roi: Box | None = None,
    mask_regions: Iterable[Box] | None = None,
    blur_regions: Iterable[Box] | None = None,
    blur_kernel_size: tuple[int, int] = (31, 31),
) -> tuple[np.ndarray, dict]:
    working_frame, roi_info = crop_roi(frame, roi=roi)
    working_frame, mask_info = apply_mask_regions(working_frame, mask_regions=mask_regions)
    working_frame, blur_info = apply_blur_regions(
        working_frame,
        blur_regions=blur_regions,
        kernel_size=blur_kernel_size,
    )

    privacy_flags = {
        **roi_info,
        **mask_info,
        **blur_info,
    }

    return working_frame, privacy_flags