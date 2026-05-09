#!/usr/bin/env python3
"""
Optimized pipeline: all fixation variants, parallel filter execution, reports every batch.
"""
import argparse
import io
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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


def vqa_accuracy(predicted, answers):
    return min(sum(1 for a in answers if a["answer"].lower() == predicted.lower()) / 3.0, 1.0)


def print_table(n_images, vqa_accs, det_preds, coco_gt):
    print(f"\n{'='*64}", flush=True)
    print(f"  Results after {n_images} images  [{time.strftime('%H:%M:%S')}]", flush=True)
    print(f"{'='*64}", flush=True)
    print(f"\n  VQA Accuracy — ViLT on VQAv2")
    print(f"  {'Fixation':<22} {'Accuracy':>10}  {'N questions':>12}")
    print(f"  {'-'*46}")
    for v in VARIANTS:
        accs = vqa_accs[v]
        if accs:
            print(f"  {v:<22} {np.mean(accs)*100:>9.2f}%  {len(accs):>12}")
        else:
            print(f"  {v:<22} {'—':>10}  {'0':>12}")

    print(f"\n  Detection mAP — DETR on COCO val2017")
    print(f"  {'Fixation':<22} {'mAP@.5:.95':>10}  {'Detections':>12}")
    print(f"  {'-'*46}")
    for v in VARIANTS:
        preds = det_preds[v]
        if preds:
            try:
                coco_dt = coco_gt.loadRes(preds)
                ev = COCOeval(coco_gt, coco_dt, "bbox")
                ev.params.imgIds = list({p["image_id"] for p in preds})
                ev.evaluate()
                ev.accumulate()
                old = sys.stdout; sys.stdout = io.StringIO()
                ev.summarize()
                sys.stdout = old
                print(f"  {v:<22} {ev.stats[0]*100:>9.2f}%  {len(preds):>12}")
            except Exception:
                print(f"  {v:<22} {'(too few yet)':>10}  {len(preds):>12}")
        else:
            print(f"  {v:<22} {'—':>10}  {'0':>12}")
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

    print("Loading COCO annotations...", flush=True)
    coco_gt = COCO(args.annotations)

    print("Loading VQA data...", flush=True)
    with open(args.vqa_questions) as f:
        vqa_q = json.load(f)
    with open(args.vqa_annotations) as f:
        vqa_ann = json.load(f)
    answer_lookup = {a["question_id"]: a["answers"] for a in vqa_ann["annotations"]}
    questions_by_image = defaultdict(list)
    for q in vqa_q["questions"]:
        questions_by_image[q["image_id"]].append({
            "question_id": q["question_id"],
            "question": q["question"],
            "answers": answer_lookup.get(q["question_id"], []),
        })

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

    vqa_accs  = defaultdict(list)
    det_preds = defaultdict(list)
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

                for det in detr.predict(arr):
                    det_preds[vname].append({
                        "image_id": image_id,
                        "category_id": det["label"],
                        "bbox": det["bbox"],
                        "score": det["score"],
                    })
                for q in questions_by_image.get(image_id, []):
                    predicted = vilt.predict(arr, q["question"])
                    vqa_accs[vname].append(vqa_accuracy(predicted, q["answers"]))

        n_processed += len(batch)
        print_table(n_processed, vqa_accs, det_preds, coco_gt)

        with open(results_dir / "partial_results.json", "w") as f:
            json.dump({
                "n_images": n_processed,
                "vqa":  {k: {"accuracy": float(np.mean(v)), "n": len(v)} for k, v in vqa_accs.items() if v},
                "det":  {k: {"n_detections": len(v)} for k, v in det_preds.items()},
            }, f, indent=2)


if __name__ == "__main__":
    main()
