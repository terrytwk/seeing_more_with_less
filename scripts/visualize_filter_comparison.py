#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUTS_DIR = "/home/terrytwk/orcd/scratch/vqav2/outputs"
DEFAULT_VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_METHOD_ORDER = [
    "var_center",
    "var_deepgaze",
    "var_frcnn",
    "var_gradient",
    "var_random",
    "const",
]
FILTERED_DIR_RE = re.compile(r"^(?P<method>.+)_\d+d_\d+perc$")
FILTERED_IMAGE_RE = re.compile(r"^(?P<stem>.+)_oid_(?P<oid>[^_]+)_fpx_(?P<fpx>\d+)_fpy_(?P<fpy>\d+)$")
TRAILING_NUMBER_RE = re.compile(r"(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create side-by-side visual comparisons of filtered images."
    )
    parser.add_argument(
        "--outputs-dir",
        default=DEFAULT_OUTPUTS_DIR,
        help=f"Top-level outputs directory. Defaults to {DEFAULT_OUTPUTS_DIR}.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset folder under OUTPUTS_DIR/filtered, e.g. vqav2_5000img. "
            "Only needed when multiple filtered datasets exist."
        ),
    )
    parser.add_argument(
        "--image-id",
        default=None,
        help=(
            "Image stem or numeric id to visualize, e.g. "
            "COCO_val2014_000000010014 or 10014. Defaults to the first image "
            "available for all selected methods."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHOD_ORDER,
        help=(
            "Methods to compare. Accepts directory names like var_center and "
            "short names like center, deepgaze, detr, gradient, random, uniform."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to save the figure. Defaults to "
            "OUTPUTS_DIR/figures/filter_comparison_<image>.png."
        ),
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional directory of original images. When set, adds an Original column.",
    )
    parser.add_argument(
        "--no-fixation-marker",
        action="store_true",
        help="Do not draw fixation points parsed from filtered filenames.",
    )
    parser.add_argument(
        "--run-vqa",
        action="store_true",
        help="Run ViLT VQA inference for the selected image and add accuracy to each panel.",
    )
    parser.add_argument(
        "--vqa-questions",
        default=None,
        help="VQAv2 questions JSON. Defaults to OUTPUTS_DIR/../v2_OpenEnded_mscoco_val2014_questions.json.",
    )
    parser.add_argument(
        "--vqa-annotations",
        default=None,
        help="VQAv2 annotations JSON. Defaults to OUTPUTS_DIR/../v2_mscoco_val2014_annotations.json.",
    )
    parser.add_argument(
        "--vqa-model",
        default=DEFAULT_VQA_MODEL,
        help=f"ViLT model path or HuggingFace id. Defaults to {DEFAULT_VQA_MODEL}.",
    )
    parser.add_argument("--device", default="auto", help="Device for VQA inference: auto, cuda, mps, or cpu.")
    parser.add_argument(
        "--vqa-output",
        default=None,
        help="Optional JSON path for detailed per-question VQA predictions and scores.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Saved figure DPI.")
    return parser.parse_args()


def method_alias(name):
    aliases = {
        "center": "var_center",
        "deepgaze": "var_deepgaze",
        "frcnn": "var_frcnn",
        "gradient": "var_gradient",
        "random": "var_random",
        "uniform": "const",
        "constant": "const",
    }
    return aliases.get(name, name)


def pretty_method_name(method):
    labels = {
        "var_center": "center",
        "var_deepgaze": "deepgaze",
        "var_frcnn": "frcnn",
        "var_gradient": "gradient",
        "var_random": "random",
        "const": "uniform",
    }
    return labels.get(method, method)


def is_filtered_root(path):
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_dir() or not FILTERED_DIR_RE.match(child.name):
            continue
        if (child / "Variable").is_dir() or (child / "Constant").is_dir():
            return True
    return False


def resolve_filtered_root(outputs_dir, dataset):
    outputs_dir = Path(outputs_dir)
    if is_filtered_root(outputs_dir):
        return outputs_dir

    filtered_parent = outputs_dir / "filtered"
    if dataset:
        root = filtered_parent / dataset
        if not is_filtered_root(root):
            raise FileNotFoundError(f"No filtered methods found under {root}")
        return root

    candidates = []
    if filtered_parent.is_dir():
        candidates = [child for child in sorted(filtered_parent.iterdir()) if is_filtered_root(child)]

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No filtered dataset folders found under {outputs_dir} or {filtered_parent}"
        )

    names = ", ".join(child.name for child in candidates)
    raise ValueError(f"Multiple filtered datasets found: {names}. Pass --dataset.")


def discover_method_dirs(filtered_root):
    method_dirs = {}
    for child in sorted(filtered_root.iterdir()):
        match = FILTERED_DIR_RE.match(child.name)
        if not match:
            continue
        method = match.group("method")
        subdir = child / ("Constant" if method == "const" else "Variable")
        if not subdir.is_dir():
            alternatives = [p for p in (child / "Variable", child / "Constant") if p.is_dir()]
            if not alternatives:
                continue
            subdir = alternatives[0]
        method_dirs[method] = subdir
    return method_dirs


def filtered_stem(path):
    match = FILTERED_IMAGE_RE.match(path.stem)
    return match.group("stem") if match else path.stem


def fixation_from_path(path):
    match = FILTERED_IMAGE_RE.match(path.stem)
    if not match:
        return None
    return int(match.group("fpx")), int(match.group("fpy"))


def numeric_suffix(stem):
    match = TRAILING_NUMBER_RE.search(stem)
    if not match:
        return None
    return match.group(1).lstrip("0") or "0"


def index_images(method_dirs):
    index = {}
    for method, image_dir in method_dirs.items():
        images = {}
        for path in sorted(image_dir.glob("*.jpg")):
            images[filtered_stem(path)] = path
        index[method] = images
    return index


def select_image_stem(image_index, methods, image_id):
    available = [set(image_index[method]) for method in methods if method in image_index]
    if not available:
        raise ValueError("No selected methods have images.")

    common = set.intersection(*available)
    if image_id is None:
        if not common:
            raise ValueError("No image is present in all selected method directories.")
        return sorted(common)[0]

    requested = image_id.strip()
    requested_number = numeric_suffix(requested)
    all_stems = sorted(set.union(*available))

    for stem in all_stems:
        if stem == requested:
            return stem
    if requested_number is not None:
        for stem in all_stems:
            if numeric_suffix(stem) == requested_number:
                return stem

    raise ValueError(f"Could not find image id {image_id!r} in selected method directories.")


def find_original(images_dir, stem):
    if images_dir is None:
        return None
    images_dir = Path(images_dir)
    candidates = [
        images_dir / f"{stem}.jpg",
        images_dir / f"{stem}.jpeg",
        images_dir / f"{stem}.png",
    ]
    suffix = numeric_suffix(stem)
    if suffix is not None:
        candidates.extend(sorted(images_dir.glob(f"*{suffix.zfill(12)}.*")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def default_output_path(outputs_dir, stem):
    return Path(outputs_dir) / "figures" / f"filter_comparison_{stem}.png"


def default_vqa_questions_path(outputs_dir):
    return Path(outputs_dir).resolve().parent / "v2_OpenEnded_mscoco_val2014_questions.json"


def default_vqa_annotations_path(outputs_dir):
    return Path(outputs_dir).resolve().parent / "v2_mscoco_val2014_annotations.json"


def questions_for_image(questions_path, annotations_path, image_id):
    from evaluation.vqa import load_vqav2

    return [
        question
        for question in load_vqav2(questions_path, annotations_path)
        if question["image_id"] == image_id
    ]


def run_vqa_for_panels(panels, image_id, questions_path, annotations_path, model_path, device):
    import numpy as np
    from PIL import Image

    from evaluation.vqa import normalize_answer, vqa_accuracy
    from inference.vilt.vilt_runner import ViLTRunner

    questions = questions_for_image(questions_path, annotations_path, image_id)
    if not questions:
        raise ValueError(f"No VQAv2 questions found for image id {image_id}.")

    runner = ViLTRunner(model_path=model_path, device=device)
    all_results = []

    for panel in panels:
        with Image.open(panel["path"]) as img:
            image = np.asarray(img.convert("RGB"))

        scores = []
        panel_results = []
        for question in questions:
            predicted = runner.predict(image, question["question"])
            accuracy = vqa_accuracy(predicted, question["answers"])
            scores.append(accuracy)
            panel_results.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "answer_type": question.get("answer_type"),
                    "question_type": question.get("question_type"),
                    "multiple_choice_answer": question.get("multiple_choice_answer"),
                    "predicted": predicted,
                    "normalized_predicted": normalize_answer(predicted),
                    "accuracy": accuracy,
                }
            )

        panel_accuracy = float(np.mean(scores))
        panel["vqa"] = {
            "accuracy": panel_accuracy,
            "n": len(scores),
            "results": panel_results,
        }
        panel["title"] = f"{panel['title']}\nVQA {panel_accuracy * 100:.1f}% ({len(scores)}q)"
        all_results.append(
            {
                "panel": panel["title"].split("\n", 1)[0],
                "image_path": str(panel["path"]),
                "accuracy": panel_accuracy,
                "n": len(scores),
                "results": panel_results,
            }
        )

    return {
        "image_id": image_id,
        "questions": len(questions),
        "questions_path": str(questions_path),
        "annotations_path": str(annotations_path),
        "model_path": model_path,
        "panels": all_results,
    }


def print_vqa_results(vqa_results):
    print("\nVQA image metrics")
    for panel in vqa_results["panels"]:
        print(f"  {panel['panel']:<10} {panel['accuracy'] * 100:>5.1f}% ({panel['n']} questions)")
        for result in panel["results"]:
            print(
                "    "
                f"{result['question_id']}: {result['question']} "
                f"| pred={result['predicted']!r} "
                f"| gt={result['multiple_choice_answer']!r} "
                f"| score={result['accuracy']:.3f}"
            )


def draw_comparison(panels, output_path, dpi, mark_fixations):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    n_panels = len(panels)
    fig_width = max(3.2 * n_panels, 6)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.2), squeeze=False)
    axes = axes[0]

    for ax, panel in zip(axes, panels):
        path = panel["path"]
        img = np.asarray(Image.open(path).convert("RGB"))
        ax.imshow(img)
        ax.set_title(panel["title"], fontsize=10)
        ax.axis("off")

        fixation = panel.get("fixation")
        if mark_fixations and fixation is not None:
            fx, fy = fixation
            radius = max(8, int(math.sqrt(img.shape[0] * img.shape[1]) * 0.02))
            circle = patches.Circle((fx, fy), radius=radius, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(circle)
            ax.plot(fx, fy, "r+", markersize=9, markeredgewidth=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    filtered_root = resolve_filtered_root(args.outputs_dir, args.dataset)
    discovered = discover_method_dirs(filtered_root)
    requested_methods = [method_alias(method) for method in args.methods]
    methods = [method for method in requested_methods if method in discovered]
    missing = [method for method in requested_methods if method not in discovered]

    if not methods:
        available = ", ".join(sorted(discovered)) or "none"
        raise ValueError(f"No requested methods were found. Available methods: {available}")

    image_index = index_images({method: discovered[method] for method in methods})
    stem = select_image_stem(image_index, methods, args.image_id)

    panels = []
    original = find_original(args.images_dir, stem)
    if original is not None:
        panels.append({"title": "original", "path": original})

    for method in methods:
        path = image_index[method].get(stem)
        if path is None:
            continue
        panels.append(
            {
                "title": pretty_method_name(method),
                "path": path,
                "fixation": fixation_from_path(path),
            }
        )

    vqa_results = None
    if args.run_vqa:
        image_id = int(numeric_suffix(stem))
        questions_path = Path(args.vqa_questions) if args.vqa_questions else default_vqa_questions_path(args.outputs_dir)
        annotations_path = (
            Path(args.vqa_annotations)
            if args.vqa_annotations
            else default_vqa_annotations_path(args.outputs_dir)
        )
        if not questions_path.is_file():
            raise FileNotFoundError(f"Missing VQAv2 questions file: {questions_path}")
        if not annotations_path.is_file():
            raise FileNotFoundError(f"Missing VQAv2 annotations file: {annotations_path}")
        print(f"Running VQA inference for image {image_id}...", flush=True)
        vqa_results = run_vqa_for_panels(
            panels,
            image_id=image_id,
            questions_path=questions_path,
            annotations_path=annotations_path,
            model_path=args.vqa_model,
            device=args.device,
        )

    output_path = Path(args.output) if args.output else default_output_path(args.outputs_dir, stem)
    draw_comparison(panels, output_path, args.dpi, not args.no_fixation_marker)

    print(f"Filtered root: {filtered_root}")
    print(f"Image: {stem}")
    print(f"Methods: {', '.join(pretty_method_name(method) for method in methods)}")
    if missing:
        print(f"Skipped missing methods: {', '.join(missing)}")
    if vqa_results is not None:
        print_vqa_results(vqa_results)
        if args.vqa_output:
            vqa_output_path = Path(args.vqa_output)
            vqa_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(vqa_output_path, "w") as f:
                json.dump(vqa_results, f, indent=2)
            print(f"Wrote VQA details {vqa_output_path}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
