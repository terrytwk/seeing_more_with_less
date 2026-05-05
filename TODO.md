# TODO

## Fix foveation blackout artifacts

- [ ] Handle interpolation NaNs in `sampling_schemes/filter_image.py` after `scipy.interpolate.griddata(...)` to prevent black rounded corner artifacts.
- [ ] Add a fallback interpolation pass (for example, fill NaN pixels with `nearest`) when `cubic` leaves gaps.
- [ ] Add a small debug print or counter for NaN pixel count per image to verify when/where this failure mode appears.

## Improve behavior when fixation shifts far from center

- [ ] Document and review the `max_out_img_sz` crop cap behavior in `filter_preprocessing(...)` so large images do not appear unexpectedly truncated.
- [ ] Add an option to return full-frame output by pasting filtered crop onto the original image (instead of black canvas), when `is_full_fov` is enabled.
- [ ] Add CLI flag(s) to choose outside-ROI fill policy (`black`, `original`, maybe `gray`).

## Validate and test

- [ ] Reproduce with current examples (`image.avif`, `image2.jpg`) at multiple `--fp_shift_x` values and compare before/after outputs.
- [ ] Add a lightweight regression check that output images contain no NaN-derived black holes except intentional masked regions.
- [ ] Update `README.md` with expected behavior of edge/corner regions and how fixation shifts interact with output window size.
