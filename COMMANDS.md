# Reproduction Commands

All commands run from the repo root (`seeing_more_with_less/`) with the virtual environment activated.

```bash
source venv/bin/activate
```

---

## Step 1 — Generate sampling maps

```bash
python matlab/generate_sampling_maps.py --model_index 1 --fov_index 2
```

Output: `matlab/sampling_scheme_params.mat`

---

## Step 2 — Apply foveated (variable) sampling to images

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type var \
    --path ./data/raw \
    --outfolder ./filtered_output \
    --batchsize 9999 \
    --index 1
```

Output: `filtered_output_30d_3perc/Variable/`

---

## Step 2 — Apply constant (uniform) sampling to images

```bash
python sampling_schemes/filter_image.py \
    --model_index 0 \
    --fov_index 1 \
    --type const \
    --path ./data/raw \
    --outfolder ./filtered_output \
    --batchsize 9999 \
    --index 1
```

Output: `filtered_output_30d_3perc/Constant/`

---

## Notes

- `--model_index 0` = 3% pixel budget (0-based for `filter_image.py`, 1-based for `generate_sampling_maps.py`)
- `--fov_index 1` = 30° field of view (0-based for `filter_image.py`)
- Set `--batchsize` larger than your total image count to process all images in one run
- Output folder is named automatically as `{outfolder}_{fov}d_{budget}perc/`
