#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Render the foveation strategy comparison figure.")
    parser.add_argument("--image-root", default="data/coco/val2017", help="Directory containing original images.")
    parser.add_argument(
        "--filtered-root",
        default="outputs/filtered/coco_val2017",
        help="Directory containing run_pipeline.py filtered outputs.",
    )
    parser.add_argument("--image-id", default="000000000139", help="Zero-padded COCO image id stem.")
    parser.add_argument("--output", default="outputs/figures/foveation_example.png", help="Path to save the figure.")
    args = parser.parse_args()

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    original = f"{args.image_root}/{args.image_id}.jpg"
    variants = {
        "Original": original,
        "var_center\n(paper baseline)": f"{args.filtered_root}/var_center_30d_3perc/Variable/{args.image_id}_oid_999_fpx_320_fpy_213.jpg",
        "var_random\n(our method)": f"{args.filtered_root}/var_random_30d_3perc/Variable/{args.image_id}_oid_0_fpx_495_fpy_38.jpg",
        "var_gradient\n(our method)": f"{args.filtered_root}/var_gradient_30d_3perc/Variable/{args.image_id}_oid_0_fpx_561_fpy_336.jpg",
        "var_detr\n(our method)": f"{args.filtered_root}/var_detr_30d_3perc/Variable/{args.image_id}_oid_0_fpx_80_fpy_212.jpg",
        "const/uniform\n(paper baseline)": f"{args.filtered_root}/const_30d_3perc/Constant/{args.image_id}_oid_999_fpx_320_fpy_213.jpg",
    }

    fixation_points = {
        "var_center\n(paper baseline)": (320, 213),
        "var_random\n(our method)": (495, 38),
        "var_gradient\n(our method)": (561, 336),
        "var_detr\n(our method)": (80, 212),
    }

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    fig.suptitle(
        "Foveation strategies -- 3% pixel budget\n(high resolution at fixation point, blurry periphery)",
        fontsize=13,
        y=1.02,
    )

    for ax, (title, path) in zip(axes, variants.items()):
        img = np.asarray(Image.open(path).convert("RGB"))
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        if title in fixation_points:
            fx, fy = fixation_points[title]
            circle = patches.Circle((fx, fy), radius=18, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(circle)
            ax.plot(fx, fy, "r+", markersize=10, markeredgewidth=2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
