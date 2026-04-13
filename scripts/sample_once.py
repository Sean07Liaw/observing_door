import argparse
import json

from app.db import SessionLocal
from app.services.detector import detect_people
from app.services.event_builder import build_observation_event
from app.services.event_service import create_event
from app.services.image_service import process_image_once


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Box must be in format x1,y1,x2,y2")

    try:
        x1, y1, x2, y2 = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Box values must be integers") from exc

    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one end-to-end sample: image pipeline -> detector -> event creation."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Path to an input image")
    source_group.add_argument("--camera", type=int, help="Camera index, e.g. 0")

    parser.add_argument(
        "--camera-id",
        type=str,
        required=True,
        help="Camera ID to store in the event, e.g. lab-door-cam-01",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Logical zone name, e.g. doorway",
    )

    parser.add_argument(
        "--roi",
        type=parse_box,
        default=None,
        help="ROI crop box x1,y1,x2,y2",
    )
    parser.add_argument(
        "--mask",
        type=parse_box,
        action="append",
        default=None,
        help="Mask region x1,y1,x2,y2; can be repeated",
    )
    parser.add_argument(
        "--blur",
        type=parse_box,
        action="append",
        default=None,
        help="Blur region x1,y1,x2,y2; can be repeated",
    )

    parser.add_argument(
        "--save-debug-image",
        action="store_true",
        help="Save an annotated detection debug image",
    )
    parser.add_argument(
        "--detector-mode",
        type=str,
        choices=["hog", "yolo"],
        default="hog",
        help="Detection backend to use",
    )
    parser.add_argument(
        "--yolo-model-name",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or local model path",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=48,
        help="Minimum detection width to keep",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=96,
        help="Minimum detection height to keep",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.2,
        help="Minimum detection confidence to keep",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=48 * 96,
        help="Minimum detection area to keep",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=5.0,
        help="Maximum height/width ratio to keep",
    )
    parser.add_argument(
        "--unknown-confidence-threshold",
        type=float,
        default=0.3,
        help="Below this max confidence, mark scene_state as unknown when detections exist",
    )

    args = parser.parse_args()

    image_result = process_image_once(
        image_path=args.image,
        camera_index=args.camera,
        roi=args.roi,
        mask_regions=args.mask,
        blur_regions=args.blur,
    )

    detection_result = detect_people(
        image_result["output_path"],
        detector_mode=args.detector_mode,
        save_debug=args.save_debug_image,
        min_width=args.min_width,
        min_height=args.min_height,
        min_confidence=args.min_confidence,
        min_area=args.min_area,
        max_aspect_ratio=args.max_aspect_ratio,
        unknown_confidence_threshold=args.unknown_confidence_threshold,
        yolo_model_name=args.yolo_model_name,
    )

    event_in = build_observation_event(
        camera_id=args.camera_id,
        zone=args.zone,
        image_result=image_result,
        detection_result=detection_result,
    )

    db = SessionLocal()
    try:
        created_event = create_event(db, event_in)
    finally:
        db.close()

    result = {
        "image_result": image_result,
        "detection_result": detection_result,
        "created_event": {
            "id": created_event.id,
            "camera_id": created_event.camera_id,
            "zone": created_event.zone,
            "event_type": created_event.event_type,
            "occupancy_state": created_event.occupancy_state,
            "person_count_estimate": created_event.person_count_estimate,
            "confidence": created_event.confidence,
            "image_uri": created_event.image_uri,
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()