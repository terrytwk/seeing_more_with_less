#!/usr/bin/env python3
# NOTE: VQAv2 uses COCO 2014 images (filename format: COCO_val2014_000000{id:06d}.jpg).
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.vqa import load_vqav2, normalize_answer, summarize_scores, vqa_accuracy


def candidate_image_stems(image_id):
    return [f"{image_id:012d}", f"COCO_val2014_{image_id:012d}"]


def find_image(image_root, image_id):
    image_root = Path(image_root)
    for stem in candidate_image_stems(image_id):
        exact_path = image_root / f"{stem}.jpg"
        if exact_path.exists():
            return exact_path

        filtered_matches = sorted(image_root.glob(f"{stem}_oid_*.jpg"))
        if filtered_matches:
            return filtered_matches[0]
    return None


def run_vqa(image_root, model_path, questions_path, annotations_path, device="auto",
            max_questions=None, progress_every=500, runner=None):
    if runner is None:
        from inference.vilt.vilt_runner import ViLTRunner

        runner = ViLTRunner(model_path=model_path, device=device)
    questions = load_vqav2(questions_path, annotations_path)
    if max_questions is not None:
        questions = questions[:max_questions]

    results = []
    server_results = []
    scores = []
    answer_type_scores = defaultdict(list)
    missing_images = 0

    for index, question in enumerate(questions, start=1):
        image_path = find_image(image_root, question["image_id"])
        if image_path is None:
            missing_images += 1
            continue

        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"))

        predicted = runner.predict(image, question["question"])
        normalized_prediction = normalize_answer(predicted)
        accuracy = vqa_accuracy(predicted, question["answers"])
        scores.append(accuracy)

        answer_type = question.get("answer_type") or "unknown"
        answer_type_scores[answer_type].append(accuracy)

        results.append({
            "question_id": question["question_id"],
            "image_id": question["image_id"],
            "question": question["question"],
            "answer_type": answer_type,
            "question_type": question.get("question_type"),
            "multiple_choice_answer": question.get("multiple_choice_answer"),
            "predicted": predicted,
            "normalized_predicted": normalized_prediction,
            "accuracy": accuracy,
            "image_path": str(image_path),
        })
        server_results.append({
            "question_id": question["question_id"],
            "answer": normalized_prediction,
        })

        if progress_every and len(scores) % progress_every == 0:
            print(f"  [{index}/{len(questions)}] evaluated={len(scores)} accuracy={np.mean(scores):.4f}", flush=True)

    summary = {
        **summarize_scores(scores),
        "evaluated": len(scores),
        "missing_images": missing_images,
        "total_questions": len(questions),
        "answer_types": {
            answer_type: summarize_scores(type_scores)
            for answer_type, type_scores in sorted(answer_type_scores.items())
        },
    }
    return summary, results, server_results


def main():
    parser = argparse.ArgumentParser(description="Run ViLT VQA inference on VQAv2 images.")
    parser.add_argument("--path", required=True, help="Folder containing original or filtered VQAv2 images.")
    parser.add_argument("--model_path", required=True, help="Path or HuggingFace ID for ViLT model.")
    parser.add_argument("--questions", required=True, help="VQAv2 questions JSON.")
    parser.add_argument("--annotations", required=True, help="VQAv2 annotations JSON.")
    parser.add_argument("--output", default=None, help="Path to write detailed results JSON.")
    parser.add_argument("--server_output", default=None, help="Path to write VQA server-style predictions JSON.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, or cpu.")
    parser.add_argument("--max_questions", type=int, default=None, help="Limit questions for smoke tests.")
    parser.add_argument("--progress_every", type=int, default=500, help="Progress print interval.")
    args = parser.parse_args()

    summary, results, server_results = run_vqa(
        image_root=args.path,
        model_path=args.model_path,
        questions_path=args.questions,
        annotations_path=args.annotations,
        device=args.device,
        max_questions=args.max_questions,
        progress_every=args.progress_every,
    )

    accuracy = summary["accuracy"] or 0.0
    print(f"\nEvaluated {summary['evaluated']} questions. Final VQA accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    if summary["missing_images"]:
        print(f"Skipped {summary['missing_images']} questions because their image file was missing.")
    for answer_type, answer_summary in summary["answer_types"].items():
        type_accuracy = answer_summary["accuracy"] or 0.0
        print(f"  {answer_type:<8} {type_accuracy:.4f} ({answer_summary['n']} questions)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"Wrote detailed results to {output_path}")

    if args.server_output:
        server_output_path = Path(args.server_output)
        server_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(server_output_path, "w") as f:
            json.dump(server_results, f)
        print(f"Wrote VQA server-style predictions to {server_output_path}")


if __name__ == "__main__":
    main()
