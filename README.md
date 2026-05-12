<div align="center">

# Seeing More with Less

### Human-like Representations in Vision Models

<p>
  <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Gizdov_Seeing_More_with_Less_Human-like_Representations_in_Vision_Models_CVPR_2025_paper.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-D12B2B?logo=adobeacrobatreader&logoColor=white" alt="Paper PDF">
  </a>
  <a href="https://seeingmorewithless.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://www.youtube.com/watch?v=OeOXkVduwWQ">
    <img src="https://img.shields.io/badge/Video-Presentation-FF0000?logo=youtube&logoColor=white" alt="Video">
  </a>
  <a href="https://cvpr.thecvf.com/virtual/2025/poster/33296">
    <img src="https://img.shields.io/badge/CVPR%202025-Spotlight-FFD700?logo=conventionalcommits&logoColor=black" alt="CVPR 2025 Spotlight">
  </a>
  <a href="https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202025/33296.png?t=1748816150.9752386">
    <img src="https://img.shields.io/badge/Poster-PNG-8B5CF6?logo=openaccess&logoColor=white" alt="Poster">
  </a>
</p>

**[Andrey Gizdov](https://andreygizdov.com), [Shimon Ullman](https://www.weizmann.ac.il/mathcsc/ullman/), [Daniel Harari](https://www.weizmann.ac.il/mathcsc/harari/)**

**Weizmann Institute of Science**

</div>

---

## TL;DR

> Modern vision models treat every pixel equally. The human eye does not -- it concentrates resolution at the fixation point and decreases it toward the periphery. We show that applying this **foveated (variable resolution) sampling** to leading vision architectures -- without any retraining -- **boosts accuracy by up to 2.7%** under identical pixel budgets, and that models achieve **~80% of full capability using just 3% of pixels**. Foveated inputs also induce human-like internal representations: resolution-selective neurons in CNNs and globally-acting self-attention in transformers.

---

## Abstract

Large multimodal models (LMMs) typically process visual inputs with uniform resolution across the entire field of view, leading to inefficiencies when non-critical image regions are processed as precisely as key areas. Inspired by the human visual system's foveated approach, we apply a biologically inspired method to leading architectures such as MDETR, BLIP2, InstructBLIP, LLaVA, and ViLT, and evaluate their performance with variable resolution inputs. Results show that foveated sampling boosts accuracy in visual tasks like question answering and object detection under tight pixel budgets, improving performance by up to **2.7% on GQA**, **2.1% on SEED-Bench**, and **2.0% on VQAv2** compared to uniform sampling. Furthermore, our research indicates that indiscriminate resolution increases yield diminishing returns, with models achieving up to **80% of their full capability using just 3% of the pixels**, even on complex tasks. Foveated sampling also prompts more human-like processing within models, such as neuronal selectivity and globally-acting self-attention in vision transformers.

---

## Overview

This repository contains the experiment code accompanying the paper. Our experimental setup largely utilizes existing publicly available model repositories (listed [below](#external-model-repositories)), since we primarily modify the *input* to each model via our sampling pipeline. The code here captures the **novel components** we built for our experiments:

```
.
├── matlab/                      # Step 1: Sampling map generation
│   ├── generic_inverted_pyramid_model.m
│   ├── extract_model_parameters.m
│   └── generate_sampling_maps.py
│
├── sampling_schemes/            # Step 2: Apply sampling maps to images (Python)
│   ├── filter_image.py
│   └── utils.py
│
├── bin_evaluation/              # Experiment: Position-dependent detection (Fig. 4)
│   ├── seq_run_bin_eval_norm_per_bin.py
│   ├── objects/
│   │   ├── annotation_processor_obj.py
│   │   ├── prediction_processor_obj.py
│   │   ├── misk_annotation_processor_obj.py
│   │   ├── tester_obj.py
│   │   └── logger_obj.py
│   └── utils/
│       ├── util_functions.py
│       └── build_custom.py
│
├── sample_equalized_evaluation/ # Experiment: Sample-equalized eval (Sec. 3.2)
│   ├── seq_run_eval_per_equal_samples_roi.py
│   ├── objects/
│   │   ├── trail_runner_obj.py
│   │   ├── seq_runner_drawer_obj.py
│   │   ├── annotation_processor_obj.py
│   │   ├── prediction_processor_obj.py
│   │   ├── misk_annotation_processor_obj.py
│   │   ├── recycler_obj.py
│   │   ├── tester_obj.py
│   │   └── main_logger_obj.py
│   └── utils/
│       ├── util_functions.py
│       └── build_custom.py
│
├── attention_distance/          # Experiment: Self-attention analysis (Sec. 5)
│   ├── attention_dist_plotter/
│   │   ├── sequence_runner_attn_graph_generator.py
│   │   └── objects/
│   │       └── graph_plotter.py
│   └── attention_visualizer/
│       └── sequence_runner_attn_vis.py
│
├── neuron_specialization/       # Experiment: CNN neuron selectivity (Sec. 5)
│   ├── extract_and_save_featuremaps_from_variable.py
│   ├── display_n_overlaped_predictions_and_save.py
│   └── model_utils.py
│
├── fixation/                    # Fixation point strategies (new)
│   ├── common.py
│   ├── random/
│   │   └── predict_fixations.py
│   ├── gradient/
│   │   └── predict_fixations.py
│   ├── deepgaze/
│   │   └── predict_fixations.py
│   └── detr/
│       └── predict_fixations.py
│
├── inference/                   # Model inference runners (new)
│   ├── detr/
│   │   ├── detr_runner.py
│   │   └── run_detection.py
│   ├── deepgaze/
│   │   ├── deepgaze_runner.py
│   │   └── run_deepgaze.py
│   └── vilt/
│       ├── vilt_runner.py
│       └── run_vqa.py
│
├── evaluation/                  # Pipeline evaluation aggregation (new)
│   ├── object_detection.py
│   └── vqa.py
│
├── scripts/                     # One-off utilities and visualization helpers
│   ├── download_coco.py
│   └── visualize_example.py
│
├── run_pipeline.py              # End-to-end pipeline orchestration (new)
├── COMMANDS.md                   # Reproduction command sheet
├── CHANGES.md                    # Summary of fork changes
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | >= 3.8 |
| MATLAB | R2020a+ (for sampling map generation only) |
| CUDA | >= 11.0 (recommended for GPU inference) |

### Python environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/seeing-more-with-less.git
cd seeing-more-with-less

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### COCO data

Download and extract the COCO 2017 files used by the default detection pipeline:

```bash
python scripts/download_coco.py
```

This writes to `data/coco/` and downloads `annotations_trainval2017.zip` plus `val2017.zip`. To include the much larger training split, pass `--include-train`.

### VQAv2 data

Download the VQAv2 validation files used to reproduce the paper's ViLT result:

```bash
python scripts/download_vqav2.py
```

This writes COCO `val2014` images and VQAv2 validation questions/annotations under `data/vqav2/`.

### Generated outputs

Keep downloaded datasets under `data/` and generated artifacts under `outputs/`:

| Artifact | Default location |
|---|---|
| Fixation JSONs | `outputs/fixations/<dataset>/<strategy>/` |
| DeepGaze saliency maps | `outputs/saliency/<dataset>/deepgaze/` |
| Filtered images | `outputs/filtered/<dataset>/<variant>_<fov>d_<budget>perc/` |
| Evaluation results and figures | `outputs/results/`, `outputs/figures/` |

### External model repositories

Our experiments evaluate existing pretrained models. Clone and set up the relevant repositories depending on which experiments you wish to reproduce:

| Model | Repository | Used For |
|---|---|---|
| LLaVA | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | Visual QA |
| BLIP-2 / InstructBLIP | [salesforce/LAVIS](https://github.com/salesforce/LAVIS) | Visual QA |
| ViLT | [huggingface/transformers](https://github.com/huggingface/transformers) | Visual QA |
| MDETR | [ashkamath/mdetr](https://github.com/ashkamath/mdetr) | Grounding / Detection |
| DETR | [facebookresearch/detr](https://github.com/facebookresearch/detr) | Object Detection |
| Mask R-CNN | [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) | Object Detection / Segmentation |

---

## Pipeline

The full experimental pipeline consists of three stages:

```
  ┌──────────────────────────┐     ┌─────────────────────────────┐     ┌──────────────────────────────┐
  │   1. SAMPLING MAPS       │     │   2. IMAGE FILTERING        │     │   3. EVALUATION              │
  │   (Python or MATLAB)      │────>│   (Python)                  │────>│   (Python)                   │
  │                           │     │                             │     │                              │
  │  Generate variable &      │     │  Apply sampling maps to     │     │  Run detection, VQA, or      │
  │  constant resolution      │     │  images and reconstruct     │     │  attention analysis on       │
  │  sampling maps            │     │  via interpolation          │     │  processed images            │
  └──────────────────────────┘     └─────────────────────────────┘     └──────────────────────────────┘
```

---

### Step 1 -- Generate Sampling Maps

The sampling maps are generated by the inverted pyramid model, which mimics the human retinal layout: a high-resolution foveola at the fixation center with receptive field sizes increasing linearly with eccentricity (following [Poggio et al., 2014](http://arxiv.org/abs/1406.1770)).

#### Recommended: Python

This fork includes a Python port of the MATLAB sampling-map workflow:

```bash
python matlab/generate_sampling_maps.py --model_index 1 --fov_index 2
```

This writes:

```text
matlab/sampling_scheme_params.mat
```

#### Original: MATLAB

```matlab
% Generate a sampling model with 3% pixel budget at 30-degree FOV
% Syntax: generic_inverted_pyramid_model(model_index, fov_index, img_size, SAVE_TO_FILE)
%
%   model_index : pixel budget [1..11] -> [3%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%]
%   fov_index   : field of view [1..3] -> [27°, 30°, 54°]
%   img_size    : image dimensions (default: [480, 640])

[model] = generic_inverted_pyramid_model(1, 2);
```

This generates a `.mat` file containing both the **variable** and **constant** (uniform) resolution sampling filters, coordinate mappings, and model parameters.

| Parameter | Description | Options |
|---|---|---|
| `model_index` | Pixel budget percentage | 1-11 (maps to 3%-90%) |
| `fov_index` | Field of view in degrees | 1: 27°, 2: 30°, 3: 54° |
| `img_size` | Input image resolution | Default: `[480, 640]` |
| `SAVE_TO_FILE` | Write `.mat` output | Default: `true` |

---

### Step 2 -- Apply Sampling to Images (Python)

Using the `.mat` sampling maps from Step 1, apply variable or constant resolution filtering to a dataset of images. The script samples pixels according to the map and reconstructs the full image via cubic interpolation.

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type var \
    --path data/coco/val2017 \
    --outfolder outputs/filtered/coco_val2017/var_center \
    --batchsize 200 \
    --index 1
```

| Argument | Description | Default |
|---|---|---|
| `--path` | Path to source images at full resolution | *(required)* |
| `--outfolder` | Output path prefix. The sampler appends `_<fov>d_<budget>perc/{Variable,Constant}` | `out` |
| `--fov_index` | Field of view index: `0`=27°, `1`=30°, `2`=54° | `0` |
| `--model_index` | Pixel budget index: `0`-`10` (maps to 3%-90%) | `0` |
| `--type` | Sampling type: `var` (variable) or `const` (uniform) | *(required)* |
| `--batchsize` | Number of images to process per batch | `50` |
| `--index` | Batch index (processes images `[index-1]*batchsize : index*batchsize`) | `1` |
| `--pool_threads` | Number of parallel worker threads | `1` |
| `--fp_shift_x` | Horizontal fixation point shift from center (px) | `0` |
| `--fp_shift_y` | Vertical fixation point shift from center (px) | `0` |
| `--fixation_json_root` | Directory containing per-image fixation JSONs | `outputs/fixations/<dataset>/default` |
| `--fixation_json_only` | Use JSON fixations only when present, instead of also emitting center fixation | `False` |

> **Note:** The `.mat` sampling parameter file must be accessible to `utils.py`. Update the path in `utils.py:read_mat()` to point to your local copy of the generated `.mat` file.

#### Fixation Strategies

Fixation strategies live under `fixation/`. Each strategy exposes a `predict_fixations.py` script that writes per-image JSON files in the format consumed by `filter_image.py`.

**DeepGaze IIE fixation** — runs DeepGaze IIE saliency inference and selects high-probability fixation points.

```bash
python fixation/deepgaze/predict_fixations.py \
    --path data/coco/val2017 \
    --num_fixations 1 \
    --device auto \
    --saliency_overlay_root outputs/saliency/coco_val2017/deepgaze/overlays \
    --overwrite
```

**Random fixation** — selects a uniformly random point per image. Useful as a lower-bound baseline for non-center fixation.

```bash
python fixation/random/predict_fixations.py \
    --path data/coco/val2017 \
    --seed 42
```

**Gradient fixation** — selects the pixel of maximum Sobel gradient magnitude (sharpest edge), with optional Gaussian pre-smoothing to suppress noise peaks.

```bash
python fixation/gradient/predict_fixations.py \
    --path data/coco/val2017 \
    --blur_sigma 3
```

**DETR fixation** — runs DETR on each image and fixates on the centroid of the highest-confidence detection. Falls back to image center if no objects are detected.

```bash
python fixation/detr/predict_fixations.py \
    --path data/coco/val2017 \
    --model_path facebook/detr-resnet-101
```

Pass any generated fixation JSON directory to the sampler with `--fixation_json_root` and `--fixation_json_only`:

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

For raw DeepGaze saliency inference without selecting fixation points, use the inference CLI:

```bash
python inference/deepgaze/run_deepgaze.py \
    --path data/coco/val2017 \
    --map_root outputs/saliency/coco_val2017/deepgaze/maps \
    --overlay_root outputs/saliency/coco_val2017/deepgaze/overlays
```

---

### Step 3 -- Run Experiments

After generating filtered image datasets, run the downstream experiments described below.

---

## Experiments

### Position-Dependent Detection (Bin Evaluation)

**Paper reference:** Figure 4

Evaluates how detection performance varies with object position relative to the fixation point. Objects are binned by the fraction of their segmentation mask falling within the high-resolution area (HRA), and detection metrics are computed per bin.

```bash
python bin_evaluation/seq_run_bin_eval_norm_per_bin.py \
    --model-name "variable_resnet" \
    --model-config-file /path/to/config.yaml \
    --middle-boundary 100 \
    --bin-spacing 0.05 \
    --filter-preds False \
    --perform-annotation-normalization True \
    --annotation-normalization-factor 0.9 \
    --org-annotations-location /path/to/annotations.json \
    --images-location /path/to/images \
    --org-predictions-location /path/to/predictions.pth \
    --parent-storage-location /path/to/output
```

<details>
<summary><b>Full argument reference</b></summary>

| Argument | Description | Default |
|---|---|---|
| `--model-name` | Model identifier (used for directory naming and logging) | `variable_resolution_pretrained_resnet_norm` |
| `--model-config-file` | Path to model YAML configuration file | *(see code)* |
| `--middle-boundary` | Edge size (px) of the central high-resolution square | `100` |
| `--bin-spacing` | Fraction step between resolution bins (e.g., `0.05` = 20 bins) | `0.04` |
| `--filter-preds` | Whether to filter predictions by mask logit scores (`True`/`False`) | `False` |
| `--perform-annotation-normalization` | Equalize annotation counts across bins via random subsampling | `False` |
| `--annotation-normalization-factor` | Multiplier for subsampling relative to smallest bin | `0.9` |
| `--org-annotations-location` | Path to COCO-format annotation file | *(see code)* |
| `--images-location` | Directory containing evaluation images | `-` |
| `--org-predictions-location` | Path to model predictions file (`.pth`) | *(see code)* |
| `--parent-storage-location` | Base output directory for results | *(see code)* |

</details>

---

### Sample-Equalized Evaluation

**Paper reference:** Section 3.2, Supplementary Table "Sample-equalised evaluation"

Evaluates detection on objects receiving an equal number of samples under both variable and constant schemes, ensuring a fair comparison. Supports multiple randomized trials.

```bash
python sample_equalized_evaluation/seq_run_eval_per_equal_samples_roi.py \
    --model-name variable_resnet \
    --model-config-file /path/to/config.yaml \
    --middle-boundary 100 \
    --bin-threshold 0.5 \
    --filter-preds True \
    --sample-ratio-range "0.02,0.03" \
    --annotation-normalization-factor 0.9 \
    --annotation-normalization-random-seed 10 \
    --images-location /path/to/images \
    --org-predictions-location /path/to/predictions.pth \
    --parent-storage-location /path/to/output \
    --experiment-folder-identificator exp1 \
    --num-trials 3
```

<details>
<summary><b>Full argument reference</b></summary>

| Argument | Description | Default |
|---|---|---|
| `--model-name` | Model identifier | `variable_resolution_pretrained_resnet_norm` |
| `--model-config-file` | Path to model YAML config | *(see code)* |
| `--middle-boundary` | Edge size (px) of central HR square | `100` |
| `--bin-threshold` | ROI containment threshold for including objects | `0.5` |
| `--filter-preds` | Filter prediction vs. annotation files | `False` |
| `--perform-annotation-randomization` | Randomize annotations across trials | `False` |
| `--sample-ratio-range` | Range of sample ratios (comma-separated, e.g. `"0.02,0.03"`) | `0.02,0.03` |
| `--annotation-normalization-factor` | Subsample multiplier relative to smallest bin | `0.9` |
| `--annotation-normalization-random-seed` | Random seed for reproducibility (use decile values) | `None` |
| `--images-location` | Path to evaluation images | *(see code)* |
| `--org-predictions-location` | Path to predictions file | *(see code)* |
| `--parent-storage-location` | Base output directory | *(see code)* |
| `--experiment-folder-identificator` | Suffix for experiment folder name | `ann_norm` |
| `--num-trials` | Number of randomized trials to run | `1` |

</details>

---

### Self-Attention Distance Analysis

**Paper reference:** Section 5, Point I; Figure 6

Demonstrates that variable resolution sampling induces more globally-acting self-attention in vision transformers. Includes both quantitative attention distance measurement and qualitative attention map visualization.

#### Attention Distance Computation

Computes the average attention distance across encoder layers, following [Dosovitskiy et al. (2021)](https://arxiv.org/abs/2010.11929).

```bash
python attention_distance/attention_dist_plotter/sequence_runner_attn_graph_generator.py \
    --folder-path /path/to/attention_maps \
    --suffixes-to-consider baseline equiconst variable \
    --attention-region global \
    --attention-region-boundary 0.25 0.75 \
    --visualizations-parent-dir /path/to/output \
    --num-img-to-use 100
```

| Argument | Description |
|---|---|
| `--folder-path` | Directory containing `.pt` attention map files |
| `--suffixes-to-consider` | File suffixes identifying model variants (e.g., `baseline equiconst variable`) |
| `--attention-region` | Region for distance calculation: `global`, `center`, `periphery`, or `periphery_strip` |
| `--attention-region-boundary` | Boundary values for center/periphery regions (relative coordinates) |
| `--visualizations-parent-dir` | Output directory for generated plots |
| `--num-img-to-use` | Number of images to include in the analysis |

#### Attention Map Visualization

Generates side-by-side encoder and decoder attention visualizations for DETR-based models (adapted from [MDETR](https://github.com/ashkamath/mdetr)).

```bash
python attention_distance/attention_visualizer/sequence_runner_attn_vis.py \
    --model_paths /path/to/checkpoint1 /path/to/checkpoint2 \
    --dataset_paths /path/to/dataset1 /path/to/dataset2 \
    --output_path /path/to/output \
    --attention_fixation static
```

| Argument | Description |
|---|---|
| `--model_paths` | Paths to DETR model checkpoints (one per model variant) |
| `--dataset_paths` | Paths to corresponding image datasets |
| `--output_path` | Directory for saving visualizations |
| `--attention_fixation` | Fixation strategy: `static` (fixed corners) or `dynamic` (object centers) |

---

### Neuron Resolution Selectivity

**Paper reference:** Section 5, Point II; Supplementary "Human-like representations"

Extracts CNN feature maps during inference on variable resolution images, demonstrating that individual neurons develop selectivity for high- vs. low-resolution regions. The extracted tensors are used for subsequent statistical testing (permutation tests, p < 10<sup>-3</sup>).

```bash
python neuron_specialization/extract_and_save_featuremaps_from_variable.py \
    --config_file /path/to/config.yaml \
    --tensors_save_dir /path/to/save/tensors \
    --weight_dir /path/to/model/weights \
    --eval_images_folder /path/to/images \
    --eval_images_annotation /path/to/annotations.json
```

| Argument | Description |
|---|---|
| `--config_file` | Mask R-CNN model configuration file (YAML) |
| `--tensors_save_dir` | Output directory for extracted feature map tensors |
| `--weight_dir` | Directory containing trained model weights |
| `--eval_images_folder` | Directory of evaluation images |
| `--eval_images_annotation` | COCO-format annotation file for the evaluation set |

> **Setup:** Ensure `maskrcnn_benchmark` and related custom modules are on your `PYTHONPATH`. See the [Mask R-CNN repository](https://github.com/facebookresearch/maskrcnn-benchmark) for installation instructions.

## Inference & Evaluation

Lightweight HuggingFace-based inference runners for DETR (object detection) and ViLT (visual question answering). These operate on any folder of images produced by `filter_image.py`.

### DETR Object Detection

```bash
python inference/detr/run_detection.py \
    --path outputs/filtered/coco_val2017/var_center_30d_3perc/Variable \
    --model_path facebook/detr-resnet-101 \
    --output outputs/results/coco_val2017/detections_var_center.json \
    --annotations data/coco/annotations/instances_val2017.json
```

Outputs a COCO-format predictions JSON and optionally evaluates mAP via pycocotools when `--annotations` is provided.

### ViLT Visual Question Answering

```bash
python inference/vilt/run_vqa.py \
    --path outputs/filtered/vqav2_val2014/var_center_30d_3perc/Variable \
    --model_path dandelin/vilt-b32-finetuned-vqa \
    --questions data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations data/vqav2/v2_mscoco_val2014_annotations.json \
    --output outputs/results/vqav2_val2014/vqa_var_center.json
```

Reports official-style VQA soft accuracy, answer-type breakdowns, and overall accuracy.

### Reproduce ViLT VQAv2 Paper Result

The paper's ViLT result uses VQAv2 validation on COCO `val2014`, without fine-tuning ViLT, comparing full-resolution images to center-foveated variable 3% sampling and uniform 3% sampling:

```bash
python scripts/reproduce_vilt_vqav2.py \
    --images data/vqav2/val2014 \
    --questions data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations data/vqav2/v2_mscoco_val2014_annotations.json \
    --model_path dandelin/vilt-b32-finetuned-vqa
```

The expected paper-scale accuracies are approximately 81.1% full resolution, 64.9% variable 3%, and 62.9% uniform 3%. Results are written to `outputs/results/vqav2_val2014/vilt_vqav2_reproduction.json`.

---

## End-to-End Pipeline

`run_pipeline.py` ties the full workflow together: generates fixation JSONs for all strategies, runs all filter variants in parallel, runs DETR and ViLT inference, and prints a results table after every batch. Evaluation aggregation lives in `evaluation/object_detection.py` and `evaluation/vqa.py`.

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

To run detection only (skips ViLT — roughly 2× faster per batch):

```bash
python run_pipeline.py \
    --images data/coco/val2017 \
    --annotations data/coco/annotations/instances_val2017.json \
    --detr_model facebook/detr-resnet-101 \
    --batch_size 10 \
    --skip_vqa
```

Results are printed after each batch and accumulated in `outputs/results/coco_val2017/partial_results.json`. The variants compared are:

| Variant | Description |
|---|---|
| `var_center` | Foveated, fixation at image center (paper baseline) |
| `var_random` | Foveated, random fixation point |
| `var_gradient` | Foveated, fixation at peak gradient magnitude |
| `var_frcnn` | Foveated, fixation at centroid of top Faster R-CNN detection |
| `var_deepgaze` | Foveated, fixation at top DeepGaze IIE saliency location |
| `const` | Uniform sampling, no foveal effect (paper baseline) |

To render the example comparison figure after pipeline outputs exist:

```bash
python scripts/visualize_example.py
```

To compare filtered outputs for the same image across fixation strategies, use:

```bash
python scripts/visualize_filter_comparison.py \
    --outputs-dir /home/terrytwk/orcd/scratch/vqav2/outputs \
    --image-id 01757 \
    --output /tmp/filter_comparison_01757.png
```

The script auto-discovers filtered method folders such as `var_center_30d_3perc/Variable`,
`var_deepgaze_30d_3perc/Variable`, and `const_30d_3perc/Constant`. If `--image-id` is
omitted, it selects the first image available for all selected methods. To run ViLT on the
selected image and annotate each panel with image-level VQA accuracy, add `--run-vqa`:

```bash
python scripts/visualize_filter_comparison.py \
    --image-id 01757 \
    --run-vqa \
    --vqa-output /tmp/filter_comparison_01757_vqa.json \
    --output /tmp/filter_comparison_01757_vqa.png
```

---

## Project Blog / GitHub Pages

The 6.8300 final project blog is packaged as a static GitHub Pages site under:

```text
docs/
```

To deploy it from this repository:

1. Push the `dev` branch to GitHub.
2. In GitHub, open **Settings -> Pages**.
3. Set **Source** to **Deploy from a branch**.
4. Set **Branch** to `dev` and **Folder** to `/docs`.

The project page should then be available at:

```text
https://terrytwk.github.io/seeing_more_with_less/
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@InProceedings{gizdov2025seeing,
    title     = {Seeing More with Less: Human-like Representations in Vision Models},
    author    = {Gizdov, Andrey and Ullman, Shimon and Harari, Daniel},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```

---

## License

This project is released for academic and research use. Please see the individual external model repositories for their respective licenses.

## Acknowledgements

We thank the authors and maintainers of [DETR](https://github.com/facebookresearch/detr), [MDETR](https://github.com/ashkamath/mdetr), [Mask R-CNN](https://github.com/facebookresearch/maskrcnn-benchmark), [LLaVA](https://github.com/haotian-liu/LLaVA), [LAVIS (BLIP-2 / InstructBLIP)](https://github.com/salesforce/LAVIS), and [Hugging Face Transformers (ViLT)](https://github.com/huggingface/transformers) for making their code publicly available.
