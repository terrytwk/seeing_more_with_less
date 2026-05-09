#!/usr/bin/env python3
"""
Optimized pipeline: all fixation variants, parallel filter execution, reports every batch.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluation.object_detection import ObjectDetectionEvaluation
from evaluation.vqa import VQAEvaluation
from fixation.common import dataset_name_from_path, default_fixation_root

FOV_DEG = 30
BUDGET_PERC = 3
VARIANTS = ["var_center", "var_random", "var_gradient", "var_detr", "var_deepgaze", "const"]


def compute_gradient_fixation(image, blur_sigma=3):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ksize = int(6 * blur_sigma + 1) | 1
    gray = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    y, x = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    return int(y), int(x), float(magnitude[y, x])


def write_fixation_json(path, image_shape, objects_info):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"image_size": [int(image_shape[0]), int(image_shape[1])],
                   "objects_info": objects_info}, f, indent=2)
        f.write("\n")


def launch_filter(images_dir, outfolder, filter_type, batch_index, batch_size,
                  fixation_json_root=None, fixation_json_only=False):
    cmd = [
        "venv/bin/python", "sampling_schemes/filter_image.py",
        "--type", filter_type,
        "--path", str(images_dir),
        "--outfolder", str(outfolder),
        "--model_index", "0", "--fov_index", "1",
        "--batchsize", str(batch_size),
        "--index", str(batch_index),
    ]
    if fixation_json_root:
        cmd += ["--fixation_json_root", str(fixation_json_root)]
    if fixation_json_only:
        cmd += ["--fixation_json_only"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def filtered_dir(outfolder, filter_type):
    subdir = "Variable" if filter_type == "var" else "Constant"
    return Path(f"{outfolder}_{FOV_DEG}d_{BUDGET_PERC}perc") / subdir


def find_filtered_image(filt_dir, stem):
    matches = list(Path(filt_dir).glob(f"{stem}_oid_*.jpg"))
    return matches[0] if matches else None


def print_table(n_images, vqa_evaluation, detection_evaluation):
    print(f"\n{'='*64}", flush=True)
    print(f"  Results after {n_images} images  [{time.strftime('%H:%M:%S')}]", flush=True)
    print(f"{'='*64}", flush=True)
    vqa_evaluation.print_report()
    detection_evaluation.print_report()
    print(f"\n{'='*64}\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",       default="data/coco/val2017")
    parser.add_argument("--annotations",  default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--vqa_questions",  default="data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--vqa_annotations", default="data/vqav2/v2_mscoco_val2014_annotations.json")
    parser.add_argument("--detr_model",   default="data/models/detr-resnet-101")
    parser.add_argument("--vilt_model",   default="data/models/vilt-b32-finetuned-vqa")
    parser.add_argument("--outfolder",    default=None)
    parser.add_argument("--results_dir",   default=None)
    parser.add_argument("--deepgaze_centerbias", default=None)
    parser.add_argument("--deepgaze_min_distance", type=int, default=64)
    parser.add_argument("--deepgaze_threshold", type=float, default=None)
    parser.add_argument("--batch_size",   type=int, default=10)
    parser.add_argument("--max_images",   type=int, default=None)
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    images_dir = Path(args.images)
    dataset_name = dataset_name_from_path(images_dir)
    out = Path(args.outfolder) if args.outfolder else Path("outputs") / "filtered" / dataset_name
    results_dir = Path(args.results_dir) if args.results_dir else Path("outputs") / "results" / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    fp_roots = {
        "random":      str(default_fixation_root(images_dir, "random")),
        "gradient":    str(default_fixation_root(images_dir, "gradient")),
        "detr":        str(default_fixation_root(images_dir, "detr")),
        "deepgaze":    str(default_fixation_root(images_dir, "deepgaze")),
    }

    print("Loading VQA data...", flush=True)
    vqa_evaluation = VQAEvaluation(args.vqa_questions, args.vqa_annotations, VARIANTS)

    print("Loading COCO annotations...", flush=True)
    detection_evaluation = ObjectDetectionEvaluation(args.annotations, VARIANTS)

    print("Generating random fixation JSONs...", flush=True)
    subprocess.run([
        "venv/bin/python", "fixation/random/predict_fixations.py",
        "--path", str(images_dir),
        "--outfolder", fp_roots["random"],
        "--seed", "42",
    ], check=True)

    from inference.detr.detr_runner import DETRRunner
    from inference.deepgaze.deepgaze_runner import DeepGazeIIERunner
    from inference.vilt.vilt_runner import ViLTRunner
    from fixation.deepgaze.fixation_selector import select_fixation_points

    print("Loading DETR...", flush=True)
    detr = DETRRunner(model_path=args.detr_model, device=args.device)
    print("Loading DeepGaze...", flush=True)
    deepgaze = DeepGazeIIERunner(device=args.device)
    print("Loading ViLT...", flush=True)
    vilt = ViLTRunner(model_path=args.vilt_model, device=args.device)

    print("Listing images...", flush=True)
    images = sorted(images_dir.glob("*.jpg"))
    if args.max_images:
        images = images[:args.max_images]
    print(f"Found {len(images)} images. Starting batches of {args.batch_size}.\n", flush=True)

    variant_cfg = {
        "var_center":      ("var",   out / "var_center",      None,                    False),
        "var_random":      ("var",   out / "var_random",      fp_roots["random"],      True),
        "var_gradient":    ("var",   out / "var_gradient",    fp_roots["gradient"],    True),
        "var_detr":        ("var",   out / "var_detr",        fp_roots["detr"],        True),
        "var_deepgaze":    ("var",   out / "var_deepgaze",    fp_roots["deepgaze"],    True),
        "const":           ("const", out / "const",           None,                    False),
    }

    n_processed = 0

    for batch_start in range(0, len(images), args.batch_size):
        batch_num = batch_start // args.batch_size + 1
        batch = images[batch_start:batch_start + args.batch_size]
        print(f"[Batch {batch_num}] Generating gradient + DETR + DeepGaze fixation JSONs...", flush=True)

        grad_root = Path(fp_roots["gradient"])
        detr_root = Path(fp_roots["detr"])
        deepgaze_root = Path(fp_roots["deepgaze"])
        grad_root.mkdir(parents=True, exist_ok=True)
        detr_root.mkdir(parents=True, exist_ok=True)
        deepgaze_root.mkdir(parents=True, exist_ok=True)

        for img_path in batch:
            with Image.open(img_path) as img:
                h, w = img.size[1], img.size[0]
                arr = np.asarray(img.convert("RGB"))

            gjson = grad_root / img_path.with_suffix(".json").name
            if not gjson.exists():
                gy, gx, gscore = compute_gradient_fixation(arr)
                write_fixation_json(gjson, arr.shape,
                                    [{"obj_id": 0, "centroid": [gy, gx], "score": gscore}])

            detr_json = detr_root / img_path.with_suffix(".json").name
            if not detr_json.exists():
                dets = sorted(detr.predict(arr), key=lambda d: d["score"], reverse=True)
                if dets:
                    bx, by, bw, bh = dets[0]["bbox"]
                    cy, cx = int(by + bh / 2), int(bx + bw / 2)
                    score = dets[0]["score"]
                else:
                    cy, cx, score = h // 2, w // 2, 0.0
                write_fixation_json(detr_json, arr.shape,
                                    [{"obj_id": 0, "centroid": [cy, cx], "score": score}])

            deepgaze_json = deepgaze_root / img_path.with_suffix(".json").name
            if not deepgaze_json.exists():
                saliency = deepgaze.predict_probability(arr, centerbias_path=args.deepgaze_centerbias)
                objects_info = select_fixation_points(
                    saliency,
                    num_fixations=1,
                    min_distance=args.deepgaze_min_distance,
                    threshold=args.deepgaze_threshold,
                )
                write_fixation_json(deepgaze_json, arr.shape, objects_info)

        print(f"[Batch {batch_num}] Launching {len(variant_cfg)} filter variants in parallel...", flush=True)
        t0 = time.time()
        procs = {
            vname: launch_filter(images_dir, outf, ftype, batch_num, args.batch_size,
                                 fixation_json_root=fp_root, fixation_json_only=fp_only)
            for vname, (ftype, outf, fp_root, fp_only) in variant_cfg.items()
        }
        for vname, proc in procs.items():
            proc.wait()
        print(f"[Batch {batch_num}] Filtering done in {time.time()-t0:.1f}s. Running inference...", flush=True)

        for img_path in batch:
            image_id = int(img_path.stem)
            for vname, (ftype, outf, _, _) in variant_cfg.items():
                filt_path = find_filtered_image(filtered_dir(outf, ftype), img_path.stem)
                if filt_path is None:
                    continue
                with Image.open(filt_path) as img:
                    arr = np.asarray(img.convert("RGB"))

                detection_evaluation.add_predictions(vname, image_id, detr.predict(arr))
                vqa_evaluation.add_image_results(vname, image_id, arr, vilt)

        n_processed += len(batch)
        print_table(n_processed, vqa_evaluation, detection_evaluation)

        with open(results_dir / "partial_results.json", "w") as f:
            json.dump({
                "n_images": n_processed,
                "vqa": vqa_evaluation.to_json(),
                "det": detection_evaluation.to_json(),
            }, f, indent=2)


if __name__ == "__main__":
    main()
