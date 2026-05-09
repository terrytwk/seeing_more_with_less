# Changes from Original Main

This file tracks changes made relative to the original `main` branch of the forked Seeing More with Less repository.

Original baseline:

```text
main / origin/master: 40a2e90201cb8b5dd5bdcfbc0ed45843fd9cd854
```

---

## 1. Python Sampling Map Generation

**Added:** `matlab/generate_sampling_maps.py`

The original code generated sampling-map parameters with MATLAB. This adds a Python port of the MATLAB workflow from:

- `matlab/extract_model_parameters.m`
- `matlab/generic_inverted_pyramid_model.m`

The script writes:

```text
matlab/sampling_scheme_params.mat
```

Usage:

```bash
python matlab/generate_sampling_maps.py --model_index 1 --fov_index 2
```

---

## 2. Added MATLAB-vs-Python sampling verification

**New file:** `seeing_more_with_less/tools/verify_matlab_python_sampling.py`

Added an end-to-end verifier for checking whether the original MATLAB sampling code and the Python port generate equivalent `.mat` sampling parameter files and equivalent filtered images.

The verifier:

- runs MATLAB `generic_inverted_pyramid_model(...)` after loading the cluster MATLAB module
- runs the Python port with the same `model_index`, `fov_index`, and image size
- compares the `.mat` variables consumed by `sampling_schemes/filter_image.py`
- filters images from `data/raw`
- compares the decoded filtered outputs pixel-by-pixel
- writes generated artifacts and a JSON report under `verification_outputs/`

**Usage:**
```bash
venv/bin/python tools/verify_matlab_python_sampling.py \
    --model_index 1 \
    --fov_index 2 \
    --image_limit 1 \
    --sampling_type both
```

**Verification notes using `data/raw/image2.jpg`:**

- 3% / 27-degree FOV: `.mat` variables matched; `const` and `var` filtered images matched pixel-for-pixel.
- 3% / 30-degree FOV: `.mat` variables matched; `const` and `var` filtered images matched pixel-for-pixel.
- 3% / 54-degree FOV: original MATLAB code failed before generation because `sample_ratios` was misspelled as `samle_ratios` in `matlab/extract_model_parameters.m`.
- After fixing that MATLAB typo, 54-degree `.mat` variables matched within tolerance. The `var` filtered image matched pixel-for-pixel. The `const` filtered image had a very small residual difference: 514 differing pixels after JPEG decode, max channel difference 3. In-memory pre-JPEG comparison differed at 336 pixels, max channel difference 2.

The remaining 54-degree `const` difference appears to come from tiny MATLAB-vs-NumPy floating-point differences in generated filter coefficients/centers, around `1e-10`, which can be amplified by cubic interpolation in sparse regions.

---

## 3. Fixed MATLAB 54-degree sampling-ratio typo

**File:** `seeing_more_with_less/matlab/extract_model_parameters.m`

The original MATLAB code used `samle_ratios` instead of `sample_ratios` in the 54-degree FOV branch, causing `generic_inverted_pyramid_model(..., 3, ...)` to fail with:

```text
Unrecognized function or variable 'sample_ratios'
```

Fixed the typo so MATLAB can generate 54-degree sampling maps.

---

## 4. `filter_image.py` Portability Fixes

**Modified:** `sampling_schemes/filter_image.py`

Changes:

- Removed the hardcoded cluster output prefix (`/home/projects/bagon/userh/data/`).
- Made `--outfolder` behave as a normal path prefix.
- Fixed argparse help text containing `%`, which crashes on newer Python versions.
- Added support for external fixation JSON files.

Filtered images are written as:

```text
{outfolder}_{fov}d_{budget}perc/{Variable,Constant}/
```

External fixation options:

```text
--fixation_json_root
--fixation_json_only
```

---

## 5. DeepGaze IIE Fixation Generation

**Added:**

```text
fixation/deepgaze/predict_fixations.py
fixation/deepgaze/fixation_selector.py
inference/deepgaze/deepgaze_runner.py
inference/deepgaze/run_deepgaze.py
```

DeepGaze IIE can be used to generate saliency-based fixation JSONs for the foveated sampler.

Example:

```bash
python fixation/deepgaze/predict_fixations.py \
    --path data/coco/val2017 \
    --num_fixations 1 \
    --device auto \
    --saliency_overlay_root outputs/saliency/coco_val2017/deepgaze/overlays \
    --overwrite
```

Default outputs:

```text
outputs/fixations/<dataset>/deepgaze/
outputs/saliency/<dataset>/deepgaze/
```

Raw DeepGaze saliency inference without fixation selection:

```bash
python inference/deepgaze/run_deepgaze.py \
    --path data/coco/val2017 \
    --map_root outputs/saliency/coco_val2017/deepgaze/maps \
    --overlay_root outputs/saliency/coco_val2017/deepgaze/overlays
```

---

## 6. Generated Output Convention

Generated artifacts are organized under `outputs/`; downloaded/source datasets remain under `data/`.

```text
data/                         # downloaded/source datasets
outputs/fixations/<dataset>/<strategy>/
outputs/saliency/<dataset>/deepgaze/{npy,maps,overlays}/
outputs/filtered/<dataset>/<variant>_<fov>d_<budget>perc/
outputs/results/<dataset>/
outputs/figures/
```

Shared path helpers:

```text
fixation/common.py
```

For `data/coco/val2017`, the derived dataset name is:

```text
coco_val2017
```

---

## 7. COCO Download Helper

**Added:** `scripts/download_coco.py`

Downloads and extracts COCO 2017 annotations plus `val2017` by default:

```bash
python scripts/download_coco.py
```

Optional train split:

```bash
python scripts/download_coco.py --include-train
```

Default dataset layout:

```text
data/coco/annotations/
data/coco/val2017/
data/coco/train2017/        # only with --include-train
```

---

## 8. Documentation and Dependency Updates

**Added/updated:**

```text
README.md
COMMANDS.md
CHANGES.md
TODO.md
requirements.txt
```

The docs now include:

- COCO download instructions
- DeepGaze fixation generation
- external fixation JSON usage in `filter_image.py`
- the `data/` vs. `outputs/` storage convention
- reproduction commands for the non-Francis additions listed above
