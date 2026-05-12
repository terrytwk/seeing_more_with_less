import contextlib
import io
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class ObjectDetectionEvaluation:
    """Accumulates DETR detections and computes COCO bbox metrics."""

    def __init__(self, annotations_path, variants):
        self.variants = tuple(variants)
        self.coco_gt = COCO(annotations_path)
        self.predictions = defaultdict(list)

    def add_predictions(self, variant, image_id, detections):
        for det in detections:
            self.predictions[variant].append({
                "image_id": image_id,
                "category_id": det["label"],
                "bbox": det["bbox"],
                "score": det["score"],
            })

    def summary_rows(self):
        rows = []
        for variant in self.variants:
            preds = self.predictions[variant]
            if not preds:
                rows.append({
                    "variant": variant,
                    "map": None,
                    "detections": 0,
                    "status": "empty",
                })
                continue

            try:
                coco_dt = self.coco_gt.loadRes(preds)
                evaluator = COCOeval(self.coco_gt, coco_dt, "bbox")
                evaluator.params.imgIds = list({p["image_id"] for p in preds})
                evaluator.evaluate()
                evaluator.accumulate()
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluator.summarize()
                rows.append({
                    "variant": variant,
                    "map": float(evaluator.stats[0]),
                    "detections": len(preds),
                    "status": "ok",
                })
            except Exception:
                rows.append({
                    "variant": variant,
                    "map": None,
                    "detections": len(preds),
                    "status": "too_few",
                })
        return rows

    def print_report(self):
        print("\n  Detection mAP - DETR on COCO val2017")
        print(f"  {'Fixation':<22} {'mAP@.5:.95':>10}  {'Detections':>12}")
        print(f"  {'-'*46}")
        for row in self.summary_rows():
            if row["status"] == "ok":
                print(f"  {row['variant']:<22} {row['map']*100:>9.2f}%  {row['detections']:>12}")
            elif row["status"] == "too_few":
                print(f"  {row['variant']:<22} {'(too few yet)':>10}  {row['detections']:>12}")
            else:
                print(f"  {row['variant']:<22} {'-':>10}  {'0':>12}")

    def to_json(self):
        result = {}
        for row in self.summary_rows():
            entry = {"n_detections": row["detections"]}
            if row["map"] is not None:
                entry["map"] = row["map"]
            result[row["variant"]] = entry
        return result
