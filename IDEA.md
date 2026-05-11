# Fixing Off-Center Foveation Boundary Artifacts

## Problem

When fixation points are not near the image center, some filtered outputs show a circular boundary around the fixation point. Pixels outside that boundary are black.

This happens because the foveated sampling map is circular and fixation-centered, while the final output is a rectangular image. If the sampling field of view does not cover the farthest image corners, interpolation has no samples for those pixels. In `sampling_schemes/filter_image.py`, those undefined pixels are produced by:

```python
scipy.interpolate.griddata(..., method="cubic")
```

Outside the convex hull of sampled points, `griddata` returns `NaN`. Those `NaN` values later become `0` during `uint8` conversion, which appears as black.

## Scientific Context

The repo's MATLAB comments distinguish center fixation from arbitrary fixation:

- `27deg`: for center fixation on `640x480` images.
- `54deg`: for arbitrary fixation locations, because the image diagonal is about `800` px, or roughly `26.7deg` at about `30 px/deg`.

The current VQAv2 filtering uses `30deg`, which can be too small to cover all corners when the fixation point is far from the image center.

The foveation model follows Poggio-style eccentricity-dependent receptive fields:

- Highest resolution at the fixation-centered foveola.
- Receptive field size increases with eccentricity.
- The code uses `seconds_per_pixel = 120`, so one pixel corresponds to `2'` of visual angle.
- `var_foveola_cell_radius = seconds_per_pixel / 2`, so the finest cell radius is `60"` in this implementation.

This is close to the Marr, Poggio, and Hildreth result that the smallest foveal channel has an excitatory center diameter of about `1'20"` and corresponds well to a retinal ganglion cell receiving input from a single cone.

Relevant references:

- D. Marr, T. Poggio, and E. Hildreth. "Smallest channel in early human vision." Journal of the Optical Society of America 70(7), 868-870, 1980.
- T. Poggio, J. Mutch, and H. Isik. "Computational role of eccentricity dependent cortical magnification." CBMM Memo 017, 2014.
- Gizdov, Ullman, and Harari. "Seeing More with Less: Human-like Representations in Vision Models." CVPR 2025.

## Fix Options

### 1. Use Larger FOV for Off-Center Fixation

Switch off-center filtered outputs from `fov_index=1` / `30deg` to `fov_index=2` / `54deg`.

This is the quickest scientifically aligned fix because the original MATLAB comments already say `54deg` is intended for arbitrary fixation locations.

Tradeoff: the same nominal sample budget is distributed across a larger visual field. This may reduce effective foveal density relative to the current `30deg` outputs, so comparisons should be rerun consistently.

### 2. Compute Fixation-Specific FOV

For each image and fixation point, compute the visual angle needed to cover the farthest image corner:

```text
max_radius_px = max distance from fixation to any image corner
required_fov_deg = 2 * max_radius_px / pixels_per_degree
```

Then generate or select a sampling map that covers that FOV.

This is likely the cleanest scientific solution: the retina-centered field covers exactly the visible image extent needed for that fixation point.

Tradeoff: requires dynamic map generation or caching FOV buckets, which is more work than using existing `54deg` maps.

### 3. Keep Circular Valid Support and Mask Explicitly

If the intended representation is "only the visual field around the fixation is observed," pixels outside the circular field are genuinely unobserved. In that interpretation, black is not a bug, but it should be represented intentionally.

Possible display choices:

- Transparent alpha outside support.
- Neutral gray outside support.
- Explicit mask overlay in visualizations.

Tradeoff: for VQA/detection evaluation, a hard missing-data region may unfairly change the task, especially if the original benchmark assumes the whole image is visible.

### 4. Add Nearest-Neighbor Fallback for Undefined Interpolation

After cubic interpolation, fill `NaN` pixels with nearest-neighbor interpolation:

```python
cubic = scipy.interpolate.griddata(points, values, targets, method="cubic")
nearest = scipy.interpolate.griddata(points, values, targets, method="nearest")
cubic[np.isnan(cubic)] = nearest[np.isnan(cubic)]
```

This removes black artifacts without changing sampling maps.

Tradeoff: this is an engineering fix, not a true retinal model fix. It invents values outside sampled support. Still, nearest fallback is much better than allowing `NaN` to become black.

### 5. Use Linear/Bilinear Reconstruction Instead of Cubic

The project page describes reconstructing full-resolution frames with bilinear interpolation, while the current code uses cubic `griddata`.

Switching to linear interpolation would better match the paper description and reduce cubic overshoot/ringing.

Tradeoff: linear interpolation still cannot define pixels outside the convex hull, so it should be combined with either larger FOV or a nearest-neighbor fallback.

### 6. Pad Before Filtering

When fixation is near an image edge, pad the source image before applying the foveated sampling model.

Possible padding modes:

- Edge replication.
- Reflection.
- Neutral gray.

Tradeoff: biological interpretation is mixed. The retina does not hallucinate content beyond the camera frame. But if benchmark images are crops from larger scenes, edge or reflection padding may be less harmful than black artifacts.

## Recommended Direction

Use a two-layer fix:

1. Use a fixation-aware FOV. As a first implementation, use the existing `54deg` map for off-center fixation methods.
2. Add nearest-neighbor fallback for any remaining `NaN` pixels after interpolation.

This gives:

- Biologically motivated coverage for arbitrary fixation points.
- No artificial black regions.
- Stable outputs for center, DeepGaze, DETR, gradient, and random fixation methods.

Avoid only replacing `NaN` with black or gray. That hides the failure mode but keeps an unnatural boundary in the model input.

---

## Empirical Observations from Off-Center Fixation Experiments

We ran the pipeline across five fixation strategies on COCO val2017 + VQAv2 val using ViLT (VQA) and DETR (object detection) with frozen pretrained weights at 3% pixel budget, 30° FOV (`fov_index=1`).

The black circular boundary artifact described above is visible in the off-center filtered outputs. See `foveation_example.png` for a side-by-side comparison — `var_random` and `var_gradient` show the most pronounced artifacts since their fixation points are most likely to be far from center. This is consistent with the FOV analysis above: `30deg` is insufficient to cover image corners for arbitrary fixation locations.

**Preliminary results (340 images, 1757 VQA questions) — subject to update as pipeline continues:**

VQA Accuracy — ViLT on VQAv2:

| Fixation | Accuracy |
|---|---|
| `const` (uniform) | 63.9% |
| `var_main_object` | 62.3% |
| `var_center` (paper baseline) | 60.6% |
| `var_gradient` | 59.1% |
| `var_random` | 58.5% |

Detection mAP — DETR on COCO val2017:

| Fixation | mAP@.5:.95 |
|---|---|
| `var_main_object` | 8.72% |
| `var_gradient` | 6.72% |
| `var_center` | 5.80% |
| `var_random` | 4.13% |
| `const` | 3.58% |

**TODO: rerun above with `fov_index=2` (54°) for all off-center methods once the FOV fix is implemented, to get clean comparisons unaffected by the boundary artifact.**

### Note on `var_main_object` Circularity

`var_main_object` uses DETR to select the fixation point, and the same DETR model is used for detection evaluation. This means the fixation selection and evaluation are not independent — DETR naturally scores well when fixated on an object it already detected. The detection mAP advantage for `var_main_object` should be interpreted with this in mind. A cleaner experiment would use a separate model (e.g., a saliency model or human gaze data) for fixation selection and DETR only for evaluation.
