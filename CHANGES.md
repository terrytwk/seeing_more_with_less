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
