from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import cv2

from app.config import settings
from app.logger import get_logger
from app.services.capture import (
    capture_frame_from_camera,
    get_image_metadata,
    load_image,
)
from app.services.privacy import Box, apply_privacy_pipeline

logger = get_logger(__name__)


def ensure_processed_image_directory() -> None:
    settings.processed_image_dir.mkdir(parents=True, exist_ok=True)


def build_output_path(source_type: Literal["image", "camera"]) -> Path:
    ensure_processed_image_directory()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{source_type}_processed_{timestamp}.jpg"
    return settings.processed_image_dir / filename


def save_processed_image(frame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"Failed to save processed image: {path}")

    logger.info("Saved processed image: path=%s", path)
    return path


def process_image_once(
    *,
    image_path: str | None = None,
    camera_index: int | None = None,
    roi: Box | None = None,
    mask_regions: Iterable[Box] | None = None,
    blur_regions: Iterable[Box] | None = None,
    blur_kernel_size: tuple[int, int] = (31, 31),
    output_path: str | Path | None = None,
) -> dict:
    if bool(image_path) == bool(camera_index is not None):
        raise ValueError("Provide exactly one of image_path or camera_index")

    if image_path is not None:
        source_type: Literal["image", "camera"] = "image"
        input_path = Path(image_path)
        frame = load_image(input_path)
        source_ref = str(input_path)
    else:
        source_type = "camera"
        frame = capture_frame_from_camera(camera_index=camera_index or 0)
        source_ref = f"camera:{camera_index}"

    original_metadata = get_image_metadata(frame)

    processed_frame, privacy_flags = apply_privacy_pipeline(
        frame,
        roi=roi,
        mask_regions=mask_regions,
        blur_regions=blur_regions,
        blur_kernel_size=blur_kernel_size,
    )

    processed_metadata = get_image_metadata(processed_frame)

    final_output_path = (
        Path(output_path) if output_path is not None else build_output_path(source_type)
    )
    saved_path = save_processed_image(processed_frame, final_output_path)

    result = {
        "source_type": source_type,
        "source_ref": source_ref,
        "output_path": str(saved_path),
        "privacy_flags": privacy_flags,
        "metadata": {
            "original": original_metadata,
            "processed": processed_metadata,
        },
    }

    logger.info(
        "Image processing completed: source_type=%s source_ref=%s output_path=%s",
        source_type,
        source_ref,
        saved_path,
    )
    return result