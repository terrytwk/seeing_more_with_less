#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .deepgaze_runner import DeepGazeIIERunner
    from .fixation_selector import select_fixation_points
except ImportError:
    from deepgaze_runner import DeepGazeIIERunner
    from fixation_selector import select_fixation_points


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif"}


def iter_images(image_root):
    for path in sorted(Path(image_root).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_rgb_image(path):
    with Image.open(path) as pil_img:
        return np.asarray(pil_img.convert("RGB"))


def write_fixation_json(output_path, image_shape, objects_info):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_size": [int(image_shape[0]), int(image_shape[1])],
        "objects_info": objects_info,
    }
    with open(output_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def save_saliency_map(output_path, saliency):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saliency = np.asarray(saliency, dtype=np.float64)
    finite = np.isfinite(saliency)
    if not finite.any():
        normalized = np.zeros_like(saliency, dtype=np.uint8)
    else:
        min_value = float(np.min(saliency[finite]))
        max_value = float(np.max(saliency[finite]))
        if max_value > min_value:
            normalized = (255.0 * (saliency - min_value) / (max_value - min_value))
        else:
            normalized = np.zeros_like(saliency)
        normalized[~finite] = 0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    Image.fromarray(normalized, mode="L").save(output_path)


def save_saliency_overlay(output_path, image, saliency, alpha=0.45):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saliency = np.asarray(saliency, dtype=np.float64)
    finite = np.isfinite(saliency)
    if finite.any() and float(np.max(saliency[finite])) > float(np.min(saliency[finite])):
        saliency_norm = (saliency - np.min(saliency[finite])) / (np.max(saliency[finite]) - np.min(saliency[finite]))
    else:
        saliency_norm = np.zeros_like(saliency)
    saliency_norm[~finite] = 0

    heat = np.zeros_like(image, dtype=np.float64)
    heat[:, :, 0] = 255.0 * saliency_norm
    heat[:, :, 1] = 255.0 * np.clip(1.0 - np.abs(saliency_norm - 0.5) * 2.0, 0.0, 1.0)
    heat[:, :, 2] = 255.0 * (1.0 - saliency_norm)
    overlay = (1.0 - alpha) * image.astype(np.float64) + alpha * heat
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate fixation-point JSONs with DeepGaze IIE.")
    parser.add_argument("--path", required=True, help="Folder containing source images.")
    parser.add_argument(
        "--outfolder",
        default=None,
        help="Folder for fixation JSONs. Defaults to <path>/filtered/fixation_points.",
    )
    parser.add_argument("--num_fixations", type=int, default=1, help="Number of fixation points per image.")
    parser.add_argument(
        "--min_distance",
        type=int,
        default=64,
        help="Minimum pixel distance between selected fixation points.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional minimum saliency probability required for a selected point.",
    )
    parser.add_argument(
        "--centerbias",
        default=None,
        help="Optional .npy centerbias log-density template. Defaults to uniform centerbias.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, or cpu.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixation JSONs.")
    parser.add_argument(
        "--saliency_map_root",
        default=None,
        help="Optional directory for normalized grayscale saliency PNGs.",
    )
    parser.add_argument(
        "--saliency_overlay_root",
        default=None,
        help="Optional directory for saliency overlays on the source images.",
    )
    parser.add_argument(
        "--saliency_npy_root",
        default=None,
        help="Optional directory for raw saliency probability .npy arrays.",
    )
    args = parser.parse_args()

    image_root = Path(args.path)
    output_root = Path(args.outfolder) if args.outfolder else image_root / "filtered" / "fixation_points"

    runner = DeepGazeIIERunner(device=args.device)
    images = list(iter_images(image_root))
    if not images:
        print(f"No images found in {image_root}")
        return

    for image_path in images:
        output_path = output_root / image_path.with_suffix(".json").name
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing fixation JSON: {output_path}")
            continue

        print(f"Predicting DeepGaze IIE fixations for {image_path}")
        image = load_rgb_image(image_path)
        saliency = runner.predict_probability(image, centerbias_path=args.centerbias)
        objects_info = select_fixation_points(
            saliency,
            num_fixations=args.num_fixations,
            min_distance=args.min_distance,
            threshold=args.threshold,
        )
        write_fixation_json(output_path, image.shape, objects_info)
        if args.saliency_map_root:
            saliency_map_path = Path(args.saliency_map_root) / image_path.with_suffix(".png").name
            save_saliency_map(saliency_map_path, saliency)
        if args.saliency_overlay_root:
            overlay_path = Path(args.saliency_overlay_root) / image_path.with_suffix(".png").name
            save_saliency_overlay(overlay_path, image, saliency)
        if args.saliency_npy_root:
            npy_path = Path(args.saliency_npy_root) / image_path.with_suffix(".npy").name
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, saliency)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
