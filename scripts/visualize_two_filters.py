#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


DEFAULT_54D_DIR = (
    "/home/terrytwk/orcd/scratch/vqav2/outputs/filtered/"
    "vqav2_1000img_54d/var_deepgaze_30d_3perc/Variable"
)
DEFAULT_30D_DIR = (
    "/home/terrytwk/orcd/scratch/vqav2/outputs/filtered/"
    "vqav2_5000img/var_deepgaze_30d_3perc/Variable"
)
DEFAULT_OUTPUT = "outputs/figures/two_filter_comparison_id42.png"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRAILING_NUMBER_RE = re.compile(r"(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show one 54 degree filtered image next to one 30 degree filtered image."
    )
    parser.add_argument("--dir-54d", default=DEFAULT_54D_DIR, help="Directory for the 54 degree image.")
    parser.add_argument("--dir-30d", default=DEFAULT_30D_DIR, help="Directory for the 30 degree image.")
    parser.add_argument(
        "--image-id",
        default="42",
        help="Image id or full stem to match. Defaults to 42.",
    )
    parser.add_argument("--label-54d", default="54 degrees", help="Panel label for the 54 degree image.")
    parser.add_argument("--label-30d", default="30 degrees", help="Panel label for the 30 degree image.")
    parser.add_argument(
        "--title",
        default="Field of View Comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to save the figure. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Saved figure DPI.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving.",
    )
    return parser.parse_args()


def numeric_suffix(text):
    match = TRAILING_NUMBER_RE.search(text)
    if match is None:
        return None
    return match.group(1).lstrip("0") or "0"


def base_image_stem(path):
    return path.stem.split("_oid_", 1)[0]


def find_image(image_dir, image_id):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    requested = str(image_id).strip()
    requested_number = numeric_suffix(requested)
    matches = []

    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = base_image_stem(path)
        if stem == requested or path.stem == requested:
            matches.append(path)
            continue

        if requested_number is not None and numeric_suffix(stem) == requested_number:
            matches.append(path)

    if not matches:
        raise FileNotFoundError(f"No image matching id {image_id!r} found in {image_dir}")
    return matches[0]


def draw_comparison(left_path, right_path, left_label, right_label, title, output_path, dpi, show):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    panels = [(left_label, left_path), (right_label, right_path)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for ax, (label, path) in zip(axes[0], panels):
        image = np.asarray(Image.open(path).convert("RGB"))
        ax.imshow(image)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"54 degree image: {left_path}")
    print(f"30 degree image: {right_path}")
    print(f"Saved {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    image_54d = find_image(args.dir_54d, args.image_id)
    image_30d = find_image(args.dir_30d, args.image_id)
    draw_comparison(
        left_path=image_54d,
        right_path=image_30d,
        left_label=args.label_54d,
        right_label=args.label_30d,
        title=args.title,
        output_path=args.output,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
