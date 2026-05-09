#!/usr/bin/env python3
# NOTE: VQAv2 uses COCO 2014 images (filename format: COCO_val2014_000000{id:06d}.jpg).
# If using COCO 2017 images instead (format: {id:012d}.jpg), only questions whose
# image_id appears in the 2017 val set will be evaluated.
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .vilt_runner import ViLTRunner
except ImportError:
    from vilt_runner import ViLTRunner


def load_vqav2(questions_path, annotations_path):
    with open(questions_path) as f:
        questions_data = json.load(f)
    with open(annotations_path) as f:
        annotations_data = json.load(f)

    answer_lookup = {ann["question_id"]: ann["answers"] for ann in annotations_data["annotations"]}
    questions = [
        {
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "question": q["question"],
            "answers": answer_lookup.get(q["question_id"], []),
        }
        for q in questions_data["questions"]
    ]
    return questions


def vqa_accuracy(predicted, answers):
    # Standard VQA soft accuracy: min(count of predicted answer among annotators / 3, 1.0)
    count = sum(1 for a in answers if a["answer"].lower() == predicted.lower())
    return min(count / 3.0, 1.0)


def find_image(image_root, image_id):
    # Try COCO 2017 format first (000000000001.jpg), then 2014 format (COCO_val2014_000000000001.jpg)
    for pattern in [f"{image_id:012d}.jpg", f"COCO_val2014_{image_id:012d}.jpg"]:
        path = Path(image_root) / pattern
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Run ViLT VQA inference on foveated images.")
    parser.add_argument("--path", required=True, help="Folder containing foveated images.")
    parser.add_argument("--model_path", required=True, help="Path or HuggingFace ID for ViLT model.")
    parser.add_argument("--questions", required=True, help="VQAv2 questions JSON.")
    parser.add_argument("--annotations", required=True, help="VQAv2 annotations JSON.")
    parser.add_argument("--output", default=None, help="Path to write per-question results JSON.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, or cpu.")
    args = parser.parse_args()

    runner = ViLTRunner(model_path=args.model_path, device=args.device)
    questions = load_vqav2(args.questions, args.annotations)

    results = []
    total_acc = 0.0
    evaluated = 0

    for q in questions:
        image_path = find_image(args.path, q["image_id"])
        if image_path is None:
            continue

        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"))

        predicted = runner.predict(image, q["question"])
        acc = vqa_accuracy(predicted, q["answers"])
        total_acc += acc
        evaluated += 1

        results.append({
            "question_id": q["question_id"],
            "predicted": predicted,
            "accuracy": acc,
        })

        if evaluated % 500 == 0:
            print(f"  [{evaluated}/{len(questions)}] running accuracy: {total_acc / evaluated:.4f}")

    final_acc = total_acc / evaluated if evaluated > 0 else 0.0
    print(f"\nEvaluated {evaluated} questions. Final VQA accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"accuracy": final_acc, "evaluated": evaluated, "results": results}, f, indent=2)
        print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
