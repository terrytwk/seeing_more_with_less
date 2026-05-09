#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .detr_runner import DETRRunner
except ImportError:
    from detr_runner import DETRRunner

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(image_root):
    for path in sorted(Path(image_root).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def image_id_from_path(path):
    # COCO 2017 filenames are zero-padded 12-digit integers, e.g. 000000000001.jpg
    return int(path.stem)


def main():
    parser = argparse.ArgumentParser(description="Run DETR inference on a folder of images.")
    parser.add_argument("--path", required=True, help="Folder containing images.")
    parser.add_argument("--model_path", required=True, help="Path or HuggingFace ID for DETR model.")
    parser.add_argument("--output", required=True, help="Path to write COCO-format predictions JSON.")
    parser.add_argument("--annotations", default=None, help="COCO instances JSON for mAP evaluation.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, or cpu.")
    args = parser.parse_args()

    runner = DETRRunner(model_path=args.model_path, device=args.device, threshold=args.threshold)

    images = list(iter_images(args.path))
    if not images:
        print(f"No images found in {args.path}")
        return

    predictions = []
    for image_path in images:
        image_id = image_id_from_path(image_path)
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"))
        detections = runner.predict(image)
        for det in detections:
            predictions.append({
                "image_id": image_id,
                "category_id": det["label"],
                "bbox": det["bbox"],
                "score": det["score"],
            })
        print(f"  {image_path.name}: {len(detections)} detections")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(predictions, fp)
    print(f"\nWrote {len(predictions)} detections to {output_path}")

    if args.annotations:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(args.annotations)
        coco_dt = coco_gt.loadRes(str(output_path))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == "__main__":
    main()
