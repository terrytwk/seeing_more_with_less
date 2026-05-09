#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
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


def gradient_fixation_points(image, num_fixations=1, min_distance=64, blur_sigma=3):
    """
    Select fixation points at locations of maximum gradient magnitude.
    A Gaussian blur is applied before gradient computation to suppress noise peaks.
    Returns list of dicts matching the existing JSON convention (centroid is [y, x]).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if blur_sigma > 0:
        ksize = int(6 * blur_sigma + 1) | 1  # odd kernel size
        gray = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    scores = magnitude.copy()
    yy, xx = np.ogrid[:scores.shape[0], :scores.shape[1]]
    selected = []

    for obj_id in range(num_fixations):
        flat_index = int(np.argmax(scores))
        y, x = np.unravel_index(flat_index, scores.shape)
        selected.append({
            "obj_id": obj_id,
            "centroid": [int(y), int(x)],
            "score": float(magnitude[y, x]),
        })
        if min_distance == 0:
            scores[y, x] = -np.inf
        else:
            suppression_mask = (yy - y) ** 2 + (xx - x) ** 2 <= min_distance ** 2
            scores[suppression_mask] = -np.inf

    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate fixation-point JSONs using peak gradient magnitude.")
    parser.add_argument("--path", required=True, help="Folder containing source images.")
    parser.add_argument(
        "--outfolder",
        default=None,
        help="Folder for fixation JSONs. Defaults to <path>/filtered/fixation_points.",
    )
    parser.add_argument("--num_fixations", type=int, default=1, help="Number of fixation points per image.")
    parser.add_argument("--min_distance", type=int, default=64, help="Minimum pixel distance between fixation points.")
    parser.add_argument("--blur_sigma", type=float, default=3.0, help="Gaussian blur sigma before gradient (0 to disable).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixation JSONs.")
    args = parser.parse_args()

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
            image = np.asarray(img.convert("RGB"))

        objects_info = gradient_fixation_points(
            image,
            num_fixations=args.num_fixations,
            min_distance=args.min_distance,
            blur_sigma=args.blur_sigma,
        )
        write_fixation_json(output_path, image.shape, objects_info)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
