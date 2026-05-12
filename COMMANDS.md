# Reproduction Commands

All commands run from the repo root with the virtual environment activated:

```bash
source venv/bin/activate
```

Generated artifacts should go under `outputs/`. Downloaded/source datasets should stay under `data/`.

---

## 0. Download VQAv2 for ViLT Reproduction

The paper's ViLT result uses VQAv2 validation questions on COCO `val2014` images.

```bash
python scripts/download_vqav2.py
```

Outputs:

```text
data/vqav2/val2014/
data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json
data/vqav2/v2_mscoco_val2014_annotations.json
```

## 0b. Download COCO 2017

Default download: annotations plus `val2017`.

```bash
python scripts/download_coco.py
```

Optional full training split:

```bash
python scripts/download_coco.py --include-train
```

Outputs:

```text
data/coco/annotations/
data/coco/val2017/
data/coco/train2017/        # only with --include-train
```

---

## 1. Generate Sampling Maps

```bash
python matlab/generate_sampling_maps.py --model_index 1 --fov_index 2
```

Output:

```text
matlab/sampling_scheme_params.mat
```

Notes:

- `generate_sampling_maps.py` uses 1-based indices.
- `sampling_schemes/filter_image.py` uses 0-based indices.

---

## 1b. Reproduce Paper Table 1: ViLT on VQAv2

This is the focused reproduction path for the paper's ViLT/VQAv2 result. It evaluates the original pretrained ViLT checkpoint on:

- full-resolution COCO `val2014`,
- center-foveated variable sampling at 3% density,
- uniform sampling at 3% density.

```bash
python scripts/reproduce_vilt_vqav2.py \
    --images data/vqav2/val2014 \
    --questions data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations data/vqav2/v2_mscoco_val2014_annotations.json \
    --model_path dandelin/vilt-b32-finetuned-vqa
```

Expected paper-scale results:

| Condition | Accuracy |
|---|---:|
| Full resolution | ~81.1% |
| Variable 3% | ~64.9% |
| Uniform 3% | ~62.9% |

For a quick smoke test:

```bash
python scripts/reproduce_vilt_vqav2.py --conditions full --max_questions 20
```

Default outputs:

```text
outputs/filtered/vqav2_val2014/var_center_30d_3perc/Variable/
outputs/filtered/vqav2_val2014/uniform_30d_3perc/Constant/
outputs/results/vqav2_val2014/vilt_vqav2_reproduction.json
```

---

## 2. Generate Fixation JSONs

Default fixation outputs follow:

```text
outputs/fixations/<dataset>/<strategy>/
```

For COCO val2017, `<dataset>` is `coco_val2017`.

### Random

```bash
python fixation/random/predict_fixations.py \
    --path data/coco/val2017 \
    --seed 42
```

Output:

```text
outputs/fixations/coco_val2017/random/
```

### Gradient

```bash
python fixation/gradient/predict_fixations.py \
    --path data/coco/val2017 \
    --blur_sigma 3
```

Output:

```text
outputs/fixations/coco_val2017/gradient/
```

### Faster R-CNN

```bash
python fixation/faster_rcnn/predict_fixations.py \
    --path data/coco/val2017
```

Output:

```text
outputs/fixations/coco_val2017/frcnn/
```

### DeepGaze IIE

```bash
python fixation/deepgaze/predict_fixations.py \
    --path data/coco/val2017 \
    --num_fixations 1 \
    --device auto \
    --saliency_overlay_root outputs/saliency/coco_val2017/deepgaze/overlays \
    --overwrite
```

Output:

```text
outputs/fixations/coco_val2017/deepgaze/
outputs/saliency/coco_val2017/deepgaze/overlays/
```

Raw DeepGaze saliency inference without fixation selection:

```bash
python inference/deepgaze/run_deepgaze.py \
    --path data/coco/val2017 \
    --map_root outputs/saliency/coco_val2017/deepgaze/maps \
    --overlay_root outputs/saliency/coco_val2017/deepgaze/overlays
```

Default `.npy` output:

```text
outputs/saliency/coco_val2017/deepgaze/npy/
```

---

## 3. Apply Foveated Sampling

The sampler appends `_<fov>d_<budget>perc/{Variable,Constant}` to `--outfolder`.

### Center Fixation

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type var \
    --path data/coco/val2017 \
    --outfolder outputs/filtered/coco_val2017/var_center \
    --batchsize 9999 \
    --index 1
```

Output:

```text
outputs/filtered/coco_val2017/var_center_30d_3perc/Variable/
```

### Random Fixation

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type var \
    --path data/coco/val2017 \
    --outfolder outputs/filtered/coco_val2017/var_random \
    --fixation_json_root outputs/fixations/coco_val2017/random \
    --fixation_json_only \
    --batchsize 9999 \
    --index 1
```

### Gradient Fixation

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type var \
    --path data/coco/val2017 \
    --outfolder outputs/filtered/coco_val2017/var_gradient \
    --fixation_json_root outputs/fixations/coco_val2017/gradient \
    --fixation_json_only \
    --batchsize 9999 \
    --index 1
```

### Faster R-CNN Fixation

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type var \
    --path data/coco/val2017 \
    --outfolder outputs/filtered/coco_val2017/var_frcnn \
    --fixation_json_root outputs/fixations/coco_val2017/frcnn \
    --fixation_json_only \
    --batchsize 9999 \
    --index 1
```

### DeepGaze Fixation

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type var \
    --path data/coco/val2017 \
    --outfolder outputs/filtered/coco_val2017/var_deepgaze \
    --fixation_json_root outputs/fixations/coco_val2017/deepgaze \
    --fixation_json_only \
    --batchsize 9999 \
    --index 1
```

### Constant Sampling

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type const \
    --path data/coco/val2017 \
    --outfolder outputs/filtered/coco_val2017/const \
    --batchsize 9999 \
    --index 1
```

Output:

```text
outputs/filtered/coco_val2017/const_30d_3perc/Constant/
```

---

## 4. Run Inference

### DETR Detection

```bash
python inference/detr/run_detection.py \
    --path outputs/filtered/coco_val2017/var_center_30d_3perc/Variable \
    --model_path facebook/detr-resnet-101 \
    --output outputs/results/coco_val2017/detections_var_center.json \
    --annotations data/coco/annotations/instances_val2017.json
```

### ViLT VQA

```bash
python inference/vilt/run_vqa.py \
    --path outputs/filtered/coco_val2017/var_center_30d_3perc/Variable \
    --model_path dandelin/vilt-b32-finetuned-vqa \
    --questions data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations data/vqav2/v2_mscoco_val2014_annotations.json \
    --output outputs/results/coco_val2017/vqa_var_center.json
```

---

## 5. End-to-End Pipeline

Runs fixation generation, filtering, DETR inference, and batch-level reporting for center/random/gradient/Faster R-CNN/DeepGaze/constant variants. Add `--skip_vqa` to skip ViLT inference (~2× faster). DeepGaze is skipped automatically if `deepgaze_pytorch` is not installed.

```bash
python run_pipeline.py \
    --images data/coco/val2017 \
    --annotations data/coco/annotations/instances_val2017.json \
    --vqa_questions data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
    --vqa_annotations data/vqav2/v2_mscoco_val2014_annotations.json \
    --detr_model facebook/detr-resnet-101 \
    --vilt_model dandelin/vilt-b32-finetuned-vqa \
    --batch_size 10
```

Default outputs:

```text
outputs/fixations/coco_val2017/
outputs/filtered/coco_val2017/
outputs/results/coco_val2017/partial_results.json
```

---

## 6. Visualize Example

```bash
python scripts/visualize_example.py
```

Default output:

```text
outputs/figures/foveation_example.png
```

---

## Index Notes

- `--model_index 0` in `filter_image.py` = 3% pixel budget.
- `--fov_index 1` in `filter_image.py` = 30 degree field of view.
- `filter_image.py` output folders are named automatically as `{outfolder}_{fov}d_{budget}perc/`.
