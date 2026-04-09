from pathlib import Path

import cv2
import numpy as np

from app.logger import get_logger

logger = get_logger(__name__)


def load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Failed to read image: {path}")

    logger.info("Loaded image: path=%s shape=%s", path, frame.shape)
    return frame


def capture_frame_from_camera(camera_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: index={camera_index}")

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to capture frame from camera: index={camera_index}")

        logger.info("Captured frame from camera: index=%s shape=%s", camera_index, frame.shape)
        return frame
    finally:
        cap.release()


def get_image_metadata(frame: np.ndarray) -> dict:
    height, width = frame.shape[:2]

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(frame.shape[2]) if len(frame.shape) == 3 else 1,
    }