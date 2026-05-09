#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from fixation.common import default_saliency_root
    from .deepgaze_runner import DeepGazeIIERunner
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fixation.common import default_saliency_root
    from deepgaze_runner import DeepGazeIIERunner


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif"}


def iter_images(image_root):
    for path in sorted(Path(image_root).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_rgb_image(path):
    with Image.open(path) as pil_img:
        return np.asarray(pil_img.convert("RGB"))


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
            normalized = 255.0 * (saliency - min_value) / (max_value - min_value)
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
        saliency_norm = (saliency - np.min(saliency[finite])) / (
            np.max(saliency[finite]) - np.min(saliency[finite])
        )
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
    parser = argparse.ArgumentParser(description="Run DeepGaze IIE saliency inference on a folder of images.")
    parser.add_argument("--path", required=True, help="Folder containing source images.")
    parser.add_argument(
        "--npy_root",
        default=None,
        help="Directory for raw saliency probability .npy arrays. Defaults to outputs/saliency/<dataset>/deepgaze/npy.",
    )
    parser.add_argument(
        "--map_root",
        default=None,
        help="Optional directory for normalized grayscale saliency PNGs.",
    )
    parser.add_argument("--overlay_root", default=None, help="Optional directory for saliency overlays.")
    parser.add_argument(
        "--centerbias",
        default=None,
        help="Optional .npy centerbias log-density template. Defaults to uniform centerbias.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, or cpu.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing saliency outputs.")
    args = parser.parse_args()

    image_root = Path(args.path)
    npy_root = Path(args.npy_root) if args.npy_root else default_saliency_root(image_root, "deepgaze", "npy")
    map_root = Path(args.map_root) if args.map_root else None
    overlay_root = Path(args.overlay_root) if args.overlay_root else None
    runner = DeepGazeIIERunner(device=args.device)
    images = list(iter_images(image_root))
    if not images:
        print(f"No images found in {image_root}")
        return

    for image_path in images:
        npy_path = npy_root / image_path.with_suffix(".npy").name
        if npy_path.exists() and not args.overwrite:
            print(f"Skipping existing saliency output: {npy_path}")
            continue

        print(f"Predicting DeepGaze IIE saliency for {image_path}")
        image = load_rgb_image(image_path)
        saliency = runner.predict_probability(image, centerbias_path=args.centerbias)

        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, saliency)
        if map_root:
            save_saliency_map(map_root / image_path.with_suffix(".png").name, saliency)
        if overlay_root:
            save_saliency_overlay(overlay_root / image_path.with_suffix(".png").name, image, saliency)
        print(f"Wrote {npy_path}")


if __name__ == "__main__":
    main()
