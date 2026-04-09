import argparse
import json
from pathlib import Path

from app.config import settings
from app.services.image_service import process_image_once


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Box must be in format x1,y1,x2,y2"
        )

    try:
        x1, y1, x2, y2 = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Box values must be integers"
        ) from exc

    return x1, y1, x2, y2


def build_output_path(output_dir: str | None) -> Path | None:
    if output_dir is None:
        return None

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / "processed_once.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process one image or one camera frame with privacy pipeline."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Path to an input image")
    source_group.add_argument("--camera", type=int, help="Camera index, e.g. 0")

    parser.add_argument(
        "--roi",
        type=parse_box,
        default=None,
        help="ROI crop box in format x1,y1,x2,y2",
    )
    parser.add_argument(
        "--mask",
        type=parse_box,
        action="append",
        default=None,
        help="Mask region in format x1,y1,x2,y2; can be repeated",
    )
    parser.add_argument(
        "--blur",
        type=parse_box,
        action="append",
        default=None,
        help="Blur region in format x1,y1,x2,y2; can be repeated",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(31, 31),
        help="Gaussian blur kernel size, both must be odd numbers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.processed_image_dir),
        help="Directory to save processed image",
    )

    args = parser.parse_args()

    result = process_image_once(
        image_path=args.image,
        camera_index=args.camera,
        roi=args.roi,
        mask_regions=args.mask,
        blur_regions=args.blur,
        blur_kernel_size=tuple(args.blur_kernel),
        output_path=build_output_path(args.output_dir),
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()