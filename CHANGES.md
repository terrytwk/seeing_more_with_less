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

## 2. Fixed hardcoded output path in `filter_image.py`

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

## 3. Fixed argparse help string in `filter_image.py`

**File:** `seeing_more_with_less/sampling_schemes/filter_image.py`

Python 3.14's argparse treats `%` in help strings as a format character, causing a crash on startup. Escaped `%` → `%%` in the `--model_index` help string.

---

## 4. Added DeepGaze IIE fixation generation

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

---

## 5. Downloaded COCO 2017 dataset

**Directory:** `data/coco/`

Required for object detection experiments (bin evaluation, sample-equalized evaluation, neuron specialization). Downloaded the three standard splits:

| File | Size | Status |
|---|---|---|
| `annotations_trainval2017.zip` | ~241 MB | Done |
| `val2017.zip` | ~1 GB | In progress |
| `train2017.zip` | ~18 GB | In progress |

**To unzip once complete:**
```bash
cd data/coco
unzip annotations_trainval2017.zip
unzip val2017.zip
unzip train2017.zip
```

Expected layout after unzipping:
```
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017/      # ~118k images
└── val2017/        # ~5k images
```
