#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.vilt.run_vqa import run_vqa


MODEL_INDEX_3_PERCENT = 0
FOV_INDEX_30_DEG = 1
SAMPLING_MAP_MODEL_INDEX_3_PERCENT = 1
SAMPLING_MAP_FOV_INDEX_30_DEG = 2


def run_command(cmd):
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True)


def require_file(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def ensure_sampling_maps(python_exe):
    sampling_maps = REPO_ROOT / "matlab" / "sampling_scheme_params.mat"
    if sampling_maps.exists():
        return
    run_command([
        python_exe,
        REPO_ROOT / "matlab" / "generate_sampling_maps.py",
        "--model_index",
        SAMPLING_MAP_MODEL_INDEX_3_PERCENT,
        "--fov_index",
        SAMPLING_MAP_FOV_INDEX_30_DEG,
    ])


def filtered_dir(outfolder, condition):
    subdir = "Variable" if condition == "variable" else "Constant"
    stem = "var_center" if condition == "variable" else "uniform"
    return Path(f"{outfolder / stem}_30d_3perc") / subdir


def run_filter(images, outfolder, condition, batch_size, python_exe):
    filter_type = "var" if condition == "variable" else "const"
    out_prefix = outfolder / ("var_center" if condition == "variable" else "uniform")
    run_command([
        python_exe,
        REPO_ROOT / "sampling_schemes" / "filter_image.py",
        "--type",
        filter_type,
        "--path",
        images,
        "--outfolder",
        out_prefix,
        "--model_index",
        MODEL_INDEX_3_PERCENT,
        "--fov_index",
        FOV_INDEX_30_DEG,
        "--batchsize",
        batch_size,
        "--index",
        1,
    ])
    return filtered_dir(outfolder, condition)


def write_condition_results(results_dir, condition, summary, results, server_results, write_server_outputs):
    detail_path = results_dir / f"vilt_vqav2_{condition}.json"
    with open(detail_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    server_path = None
    if write_server_outputs:
        server_path = results_dir / f"vilt_vqav2_{condition}_server_predictions.json"
        with open(server_path, "w") as f:
            json.dump(server_results, f)

    return detail_path, server_path


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's ViLT VQAv2 comparison: full, variable 3%, and uniform 3%."
    )
    parser.add_argument("--images", default="data/vqav2/val2014", help="COCO val2014 image directory.")
    parser.add_argument("--questions", default="data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--annotations", default="data/vqav2/v2_mscoco_val2014_annotations.json")
    parser.add_argument("--model_path", default="dandelin/vilt-b32-finetuned-vqa")
    parser.add_argument("--outfolder", default="outputs/filtered/vqav2_val2014")
    parser.add_argument("--results_dir", default="outputs/results/vqav2_val2014")
    parser.add_argument("--batch_size", type=int, default=1000000, help="Images per filter_image.py invocation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_questions", type=int, default=None, help="Limit questions for smoke tests.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["full", "variable", "uniform"],
        default=["full", "variable", "uniform"],
        help="Conditions to evaluate. Defaults to all paper conditions.",
    )
    parser.add_argument("--skip_filter", action="store_true", help="Evaluate existing filtered outputs without regenerating them.")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip full-resolution baseline evaluation.")
    parser.add_argument("--no_generate_sampling_maps", action="store_true", help="Do not auto-create matlab/sampling_scheme_params.mat.")
    parser.add_argument("--write_server_outputs", action="store_true", help="Also write VQA server-style prediction JSON files.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocess steps.")
    args = parser.parse_args()

    images = require_file(args.images, "COCO val2014 image directory")
    questions = require_file(args.questions, "VQAv2 validation questions")
    annotations = require_file(args.annotations, "VQAv2 validation annotations")
    outfolder = Path(args.outfolder)
    results_dir = Path(args.results_dir)
    outfolder.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_generate_sampling_maps:
        ensure_sampling_maps(args.python)

    from inference.vilt.vilt_runner import ViLTRunner

    conditions = set(args.conditions)
    image_roots = {}
    if "full" in conditions and not args.skip_baseline:
        image_roots["full"] = images

    for condition in ["variable", "uniform"]:
        if condition not in conditions:
            continue
        if args.skip_filter:
            image_roots[condition] = filtered_dir(outfolder, condition)
        else:
            image_roots[condition] = run_filter(images, outfolder, condition, args.batch_size, args.python)

    aggregate = {
        "model_path": args.model_path,
        "questions": str(questions),
        "annotations": str(annotations),
        "conditions": {},
        "paper_reference": {
            "full": 0.811,
            "variable": 0.649,
            "uniform": 0.629,
        },
    }

    runner = ViLTRunner(model_path=args.model_path, device=args.device)

    for condition, image_root in image_roots.items():
        print(f"\nEvaluating {condition}: {image_root}", flush=True)
        summary, results, server_results = run_vqa(
            image_root=image_root,
            model_path=args.model_path,
            questions_path=questions,
            annotations_path=annotations,
            device=args.device,
            max_questions=args.max_questions,
            progress_every=500,
            runner=runner,
        )
        detail_path, server_path = write_condition_results(
            results_dir,
            condition,
            summary,
            results,
            server_results,
            args.write_server_outputs,
        )
        aggregate["conditions"][condition] = {
            "image_root": str(image_root),
            "summary": summary,
            "detail_output": str(detail_path),
            "server_output": str(server_path) if server_path else None,
        }

    aggregate_path = results_dir / "vilt_vqav2_reproduction.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print("\nSummary")
    for condition, info in aggregate["conditions"].items():
        accuracy = info["summary"]["accuracy"]
        if accuracy is None:
            print(f"  {condition:<8} no evaluated questions")
        else:
            print(f"  {condition:<8} {accuracy * 100:.2f}% ({info['summary']['evaluated']} questions)")
    print(f"Wrote aggregate results to {aggregate_path}")


if __name__ == "__main__":
    main()
