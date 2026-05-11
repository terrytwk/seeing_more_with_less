#!/usr/bin/env python3
"""
Fixation point selection using Faster R-CNN (torchvision).
Uses a different model family from DETR to avoid circularity when DETR
is also used for downstream object detection evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif"}


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FasterRCNNFixationRunner:
    def __init__(self, device="auto", threshold=0.5):
        self.device = resolve_device(device)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def predict_fixations(self, image, num_fixations=1):
        """
        image: H x W x 3 numpy array (RGB) or PIL Image.
        Returns list of {"obj_id": int, "centroid": [y, x], "score": float}.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        tensor = F.to_tensor(image).to(self.device)
        outputs = self.model([tensor])[0]

        keep = outputs["scores"] >= self.threshold
        boxes = outputs["boxes"][keep].cpu()
        scores = outputs["scores"][keep].cpu()

        order = scores.argsort(descending=True)
        fixations = []
        for obj_id, idx in enumerate(order[:num_fixations].tolist()):
            x1, y1, x2, y2 = boxes[idx].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            fixations.append({
                "obj_id": obj_id,
                "centroid": [cy, cx],  # [y, x] convention
                "score": float(scores[idx]),
            })

        if not fixations:
            w, h = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
            fixations.append({"obj_id": 0, "centroid": [h // 2, w // 2], "score": 0.0})

        return fixations


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
        description="Generate fixation JSONs using Faster R-CNN object detection."
    )
    parser.add_argument("--path", required=True, help="Folder containing source images.")
    parser.add_argument("--outfolder", default=None,
                        help="Folder for fixation JSONs. Defaults to <path>/filtered/fixation_points.")
    parser.add_argument("--num_fixations", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_root = Path(args.path)
    output_root = Path(args.outfolder) if args.outfolder else image_root / "filtered" / "fixation_points"
    runner = FasterRCNNFixationRunner(device=args.device, threshold=args.threshold)

    images = list(iter_images(image_root))
    if not images:
        print(f"No images found in {image_root}")
        return

    for image_path in images:
        output_path = output_root / image_path.with_suffix(".json").name
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing: {output_path}")
            continue
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"))
        fixations = runner.predict_fixations(image, num_fixations=args.num_fixations)
        write_fixation_json(output_path, image.shape, fixations)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
