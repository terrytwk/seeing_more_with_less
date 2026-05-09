# Changes to Forked Repo

Tracking modifications made to [seeingmorewithless.github.io](https://seeingmorewithless.github.io/) / CVPR 2025 codebase.

---

## 1. Python port of MATLAB sampling map generator

**New file:** `seeing_more_with_less/matlab/generate_sampling_maps.py`

The original Step 1 of the pipeline required MATLAB (R2020a+) to generate the sampling parameter file. This script is a full Python port of both MATLAB files:

- `matlab/extract_model_parameters.m` — model parameter setup via cubic spline interpolation
- `matlab/generic_inverted_pyramid_model.m` — foveated + uniform Gaussian filter generation

The script produces `matlab/sampling_scheme_params.mat` in HDF5 v7.3 format, which is read directly by `sampling_schemes/filter_image.py` via `hdf5storage`.

**Usage:**
```bash
pip install hdf5storage
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

## 4. Fixed hardcoded output path in `filter_image.py`

**File:** `seeing_more_with_less/sampling_schemes/filter_image.py`

The `--outfolder` argument was prepended with a hardcoded Linux cluster path (`/home/projects/bagon/userh/data/`), making the script fail outside that environment.

**Before:**
```python
new_database_path = os.path.join("/home/projects/bagon/userh/data/", args.outfolder + ...)
```

**After:**
```python
new_database_path = os.path.join(args.outfolder + ...)
```

`--outfolder` now behaves as a plain path prefix relative to the working directory.

---

## 5. Fixed argparse help string in `filter_image.py`

**File:** `seeing_more_with_less/sampling_schemes/filter_image.py`

Python 3.14's argparse treats `%` in help strings as a format character, causing a crash on startup. Escaped `%` → `%%` in the `--model_index` help string.

---

## 6. Added DeepGaze IIE fixation generation

**New directory:** `seeing_more_with_less/saliency/deepgaze/`
**Updated file:** `seeing_more_with_less/sampling_schemes/filter_image.py`

Generate saliency-based fixation JSONs, with optional overlay output for visual inspection:

```bash
saliency/deepgaze/predict_fixations.py \
    --path ./data/raw \
    --num_fixations 1 \
    --device auto \
    --saliency_overlay_root ./data/raw/filtered/deepgaze_saliency_overlays \
    --overwrite
```

`filter_image.py` now accepts externally generated fixation JSONs:

- `--fixation_json_root`: directory containing per-image fixation JSONs. Defaults to `<path>/filtered/fixation_points`, preserving the previous behavior.
- `--fixation_json_only`: when a JSON is present, use only JSON fixation points instead of also generating the default center fixation.

Example foveation command using DeepGaze fixations:

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type var \
    --path ./data/raw \
    --outfolder ./filtered_output_deepgaze \
    --fixation_json_root ./data/raw/filtered/fixation_points \
    --fixation_json_only \
    --batchsize 9999 \
    --index 1
```
