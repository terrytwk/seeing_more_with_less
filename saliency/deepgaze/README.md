# DeepGaze IIE Fixation Points

This directory generates fixation-point JSON files with DeepGaze IIE for the
existing foveated sampling pipeline.

DeepGaze IIE is used because its calibration paper reports stronger
out-of-domain behavior than models tuned around dataset-specific bias.

## Generate fixation JSONs

```bash
python saliency/deepgaze/predict_fixations.py \
    --path ./data/raw \
    --num_fixations 1 \
    --min_distance 64 \
    --device auto \
    --saliency_map_root ./data/raw/filtered/deepgaze_saliency_maps \
    --saliency_overlay_root ./data/raw/filtered/deepgaze_saliency_overlays \
    --saliency_npy_root ./data/raw/filtered/deepgaze_saliency_npy \
    --overwrite
```

On Apple Silicon, `auto` uses MPS when available. The wrapper converts
DeepGaze IIE's internal buffers to float32 before moving the model to MPS,
because MPS does not support float64 tensors.

By default, JSON files are written to:

```text
./data/raw/filtered/fixation_points/
```

Each JSON is compatible with `sampling_schemes/filter_image.py`:

```json
{
  "image_size": [1080, 1920],
  "objects_info": [
    {"obj_id": 0, "centroid": [540, 960], "score": 0.00012}
  ]
}
```

`centroid` is stored as `[y, x]`, matching the existing object-centroid JSON
convention used by the sampler.

Optional saliency debug outputs:

```text
--saliency_map_root      normalized grayscale saliency PNGs for visual inspection
--saliency_overlay_root  source images with saliency heatmaps overlaid
--saliency_npy_root      raw saliency probability arrays used for fixation selection
```

The grayscale PNGs are min-max normalized per image for readability. The
`.npy` arrays preserve the actual probability maps produced by DeepGaze IIE;
each array has shape `(height, width)` and should sum to approximately `1.0`.
Use the `.npy` files when you want to verify the exact values used to select
fixation points.

With the command above, debug outputs are written to:

```text
./data/raw/filtered/deepgaze_saliency_maps/
./data/raw/filtered/deepgaze_saliency_overlays/
./data/raw/filtered/deepgaze_saliency_npy/
```

## Apply foveated sampling

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

## Centerbias

The default centerbias is uniform, so no extra asset is required. To use the
MIT1003 centerbias from the DeepGaze release, pass:

```bash
--centerbias /path/to/centerbias_mit1003.npy
```
