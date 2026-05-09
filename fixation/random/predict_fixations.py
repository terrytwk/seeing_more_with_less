#!/usr/bin/env python3
import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Generate fixation-point JSONs with random fixation.")
    parser.add_argument("--path", required=True, help="Folder containing source images.")
    parser.add_argument(
        "--outfolder",
        default=None,
        help="Folder for fixation JSONs. Defaults to <path>/filtered/fixation_points.",
    )
    parser.add_argument("--num_fixations", type=int, default=1, help="Number of fixation points per image.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixation JSONs.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    image_root = Path(args.path)
    output_root = Path(args.outfolder) if args.outfolder else image_root / "filtered" / "fixation_points"

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

        objects_info = [
            {
                "obj_id": obj_id,
                "centroid": [int(rng.integers(0, h)), int(rng.integers(0, w))],
                "score": 1.0,
            }
            for obj_id in range(args.num_fixations)
        ]

        write_fixation_json(output_path, (h, w), objects_info)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
