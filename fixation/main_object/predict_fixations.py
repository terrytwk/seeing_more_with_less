#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif"}


def iter_images(image_root):
    for path in sorted(Path(image_root).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def write_fixation_json(output_path, image_shape, objects_info):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_size": [int(image_shape[0]), int(image_shape[1])],
        "objects_info": objects_info,
    }
    with open(output_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixation-point JSONs using the centroid of the most confident detected object."
    )
    parser.add_argument("--path", required=True, help="Folder containing source images.")
    parser.add_argument(
        "--outfolder",
        default=None,
        help="Folder for fixation JSONs. Defaults to <path>/filtered/fixation_points.",
    )
    parser.add_argument("--model_path", default="data/models/detr-resnet-101",
                        help="Path or HuggingFace ID for DETR model.")
    parser.add_argument("--num_fixations", type=int, default=1, help="Number of fixation points per image.")
    parser.add_argument("--threshold", type=float, default=0.5, help="DETR detection confidence threshold.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, or cpu.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixation JSONs.")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "inference" / "detr"))
    from detr_runner import DETRRunner

    image_root = Path(args.path)
    output_root = Path(args.outfolder) if args.outfolder else image_root / "filtered" / "fixation_points"

    runner = DETRRunner(model_path=args.model_path, device=args.device, threshold=args.threshold)

    images = list(iter_images(image_root))
    if not images:
        print(f"No images found in {image_root}")
        return

    for image_path in images:
        output_path = output_root / image_path.with_suffix(".json").name
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing fixation JSON: {output_path}")
            continue

        with Image.open(image_path) as img:
            w, h = img.size
            image = np.asarray(img.convert("RGB"))

        detections = runner.predict(image)

        # Sort by score descending, take top num_fixations
        detections = sorted(detections, key=lambda d: d["score"], reverse=True)

        objects_info = []
        for obj_id, det in enumerate(detections[:args.num_fixations]):
            x, y, bw, bh = det["bbox"]
            cx = int(x + bw / 2)
            cy = int(y + bh / 2)
            objects_info.append({
                "obj_id": obj_id,
                "centroid": [cy, cx],  # [y, x] convention
                "score": det["score"],
            })

        if not objects_info:
            # Fall back to image center if no detections
            objects_info.append({
                "obj_id": 0,
                "centroid": [h // 2, w // 2],
                "score": 0.0,
            })

        write_fixation_json(output_path, image.shape, objects_info)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
