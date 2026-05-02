#!/usr/bin/env python3
"""
Python port of generic_inverted_pyramid_model.m and extract_model_parameters.m

Generates variable-resolution (foveated) and constant-resolution (uniform)
sampling maps and saves them to a .mat file that filter_image.py can read.

Usage:
    python generate_sampling_maps.py --model_index 1 --fov_index 2

    model_index : pixel budget  [1..11] -> [3%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%]
    fov_index   : field of view [1..3]  -> [27 deg, 30 deg, 54 deg]

Output:
    matlab/sampling_scheme_params.mat  (HDF5 v7.3 format, readable by hdf5storage)
"""

import numpy as np
import argparse
import os
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_closing
import hdf5storage
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Port of extract_model_parameters.m
# ---------------------------------------------------------------------------

def extract_model_parameters(model_index, fov_in_deg, img_size=(480, 640)):
    """
    model_index : 1-based integer (1..11)
    fov_in_deg  : one of [27, 30, 54]
    img_size    : (height, width)
    """
    model = {}
    model['img_size'] = img_size
    model['fullres_samples'] = float(img_size[0] * img_size[1])
    model['fov_in_degrees'] = fov_in_deg
    model['seconds_per_pixel'] = 120.0
    model['var_foveola_cell_radius'] = model['seconds_per_pixel'] / 2.0  # 60"

    sample_percent_names = [3, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    model['sample_percent_name'] = sample_percent_names[model_index - 1]

    if fov_in_deg == 27:
        sample_counts = [7292, 30596, 45267, 61479, 91867,
                         123660, 152192, 185280, 214268, 244219, 276859]
    elif fov_in_deg == 30:
        sample_counts = [7538, 32118, 47990, 65243, 97930,
                         132464, 163970, 200584, 233223, 266048, 303137]
    else:  # 54
        sample_counts = [9027, 41091, 62688, 86952, 136021,
                         187440, 237089, 296390, 349292, 406890, 472279]

    sample_ratios = np.array(sample_counts, dtype=float) / model['fullres_samples']
    model['sample_ratio'] = sample_ratios[model_index - 1]

    # --- var_ecc_rf_factor via cubic spline (not-a-knot, same as MATLAB spline) ---
    ecc_rf_factors = np.array([0.05, 0.0195, 0.0150, 0.0122, 0.0092,
                                0.0074, 0.0063, 0.0054, 0.0048, 0.0043, 0.00385])
    model['var_ecc_rf_factor'] = float(CubicSpline(sample_ratios, ecc_rf_factors)(model['sample_ratio']))
    model['var_max_out_img_sz'] = 1600

    # --- const_rf_radius via cubic spline ---
    const_rf_radius_vals = np.array([1010, 490, 400, 345, 280, 245, 219, 201, 185, 173, 165], dtype=float)
    model['const_rf_radius'] = float(CubicSpline(sample_ratios, const_rf_radius_vals)(model['sample_ratio']))

    # --- const_half_num_of_uniform_rf_cells via cubic spline ---
    if fov_in_deg == 27:
        const_half_cells = [48, 99, 121, 140, 171, 199, 221, 243, 262, 279, 297]
    elif fov_in_deg == 30:
        const_half_cells = [54, 111, 135, 157, 193, 221, 247, 269, 292, 313, 328]
    else:
        const_half_cells = [96, 198, 243, 282, 347, 397, 444, 484, 526, 562, 589]
    model['const_half_num_of_uniform_rf_cells'] = float(
        CubicSpline(sample_ratios, np.array(const_half_cells, dtype=float))(model['sample_ratio'])
    )

    model['const_max_out_img_sz'] = 1600
    return model


# ---------------------------------------------------------------------------
# Port of generic_inverted_pyramid_model.m
# ---------------------------------------------------------------------------

def _build_xlim_ylim(circles_cntr_xy, circles_radii, seconds_per_pixel, gauss_half_sup=50):
    """
    Compute 1-based MATLAB-style tar/src xlim/ylim for filter placement.
    Stored as 1-based because filter_image.py subtracts 1 from index [0] when slicing.
    """
    rf_roi_half_sz = int(np.floor(
        np.max(circles_cntr_xy[:, 0] / seconds_per_pixel) +
        np.max(circles_radii / seconds_per_pixel)
    ))

    n = len(circles_radii)
    tar_xlim = np.zeros((n, 2), dtype=np.float32)
    tar_ylim = np.zeros((n, 2), dtype=np.float32)
    src_xlim = np.zeros((n, 2), dtype=np.float32)
    src_ylim = np.zeros((n, 2), dtype=np.float32)

    for i in range(n):
        cx = circles_cntr_xy[i, 0] / seconds_per_pixel
        cy = circles_cntr_xy[i, 1] / seconds_per_pixel

        cx_lo = int(np.floor(cx - gauss_half_sup + 0.5))
        cx_hi = int(np.floor(cx + gauss_half_sup + 0.5))
        cy_lo = int(np.floor(cy - gauss_half_sup + 0.5))
        cy_hi = int(np.floor(cy + gauss_half_sup + 0.5))

        sz = 2 * rf_roi_half_sz + 1
        g  = 2 * gauss_half_sup + 1

        # Target window (1-based)
        tar_xlim[i, 0] = max(1, 1 + rf_roi_half_sz + cx_lo)
        tar_xlim[i, 1] = min(sz, 1 + rf_roi_half_sz + cx_hi)
        tar_ylim[i, 0] = max(1, 1 + rf_roi_half_sz + cy_lo)
        tar_ylim[i, 1] = min(sz, 1 + rf_roi_half_sz + cy_hi)

        # Source window into the 101×101 Gaussian kernel (1-based)
        src_xlim[i, 0] = max(1, 1 - (rf_roi_half_sz + cx_lo))
        src_xlim[i, 1] = min(g,  g - (rf_roi_half_sz + 1 + (cx_hi - sz)))
        src_ylim[i, 0] = max(1, 1 - (rf_roi_half_sz + cy_lo))
        src_ylim[i, 1] = min(g,  g - (rf_roi_half_sz + 1 + (cy_hi - sz)))

    return rf_roi_half_sz, tar_xlim, tar_ylim, src_xlim, src_ylim


def _build_gaussian_filters(circles_cntr_xy, circles_radii, seconds_per_pixel, gauss_half_sup=50):
    """Build a (101, 101, N) float32 array of Gaussian RF filters."""
    n = len(circles_radii)
    k = 2 * gauss_half_sup + 1
    rf_filter = np.zeros((k, k, n), dtype=np.float32)

    for i in range(n):
        cx = circles_cntr_xy[i, 0] / seconds_per_pixel
        cy = circles_cntr_xy[i, 1] / seconds_per_pixel
        sigma2 = (circles_radii[i] / seconds_per_pixel) ** 2

        cx_c = int(np.floor(cx + 0.5))
        cy_c = int(np.floor(cy + 0.5))
        xs = np.arange(cx_c - gauss_half_sup, cx_c + gauss_half_sup + 1, dtype=np.float64)
        ys = np.arange(cy_c - gauss_half_sup, cy_c + gauss_half_sup + 1, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)

        # Equivalent to MATLAB mvnpdf([X(:) Y(:)], [cx cy], eye(2)*sigma2)
        gauss = (1.0 / (2.0 * np.pi * sigma2)) * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma2))
        rf_filter[:, :, i] = gauss.astype(np.float32)

    return rf_filter


def generic_inverted_pyramid_model(model_index, fov_index,
                                   img_size=(480, 640),
                                   save_to_file=True,
                                   output_dir=None):
    """
    Python port of generic_inverted_pyramid_model.m

    model_index : 1-based [1..11] -> [3%..90%] pixel budget
    fov_index   : 1-based [1..3]  -> [27, 30, 54] degrees FOV
    img_size    : (height, width)
    """
    fov_options = [27, 30, 54]
    fov_in_deg = fov_options[fov_index - 1]
    model = extract_model_parameters(model_index, fov_in_deg, img_size)

    log(f"Generating inverted pyramid for {model['sample_percent_name']}% pixel budget "
        f"at {fov_in_deg}° FOV ...")

    spp        = model['seconds_per_pixel']          # 120"
    init_r     = model['var_foveola_cell_radius']    # 60"
    init_fac   = 2                                    # UNFILTERED_FOVEOLA = True
    gauss_half = 50

    # ------------------------------------------------------------------
    # Variable-resolution: foveola layout (concentric circles, r = fixed)
    # ------------------------------------------------------------------
    log("Building foveola cell layout ...")
    cntr = np.array([[0.0, 0.0]])
    radii = np.array([init_fac * init_r])
    concentric_r = init_fac * init_r

    for n in range(2, 21):
        concentric_r = (n - 1) * init_fac * init_r
        n_circ = int(np.ceil(360.0 / (2.0 * np.degrees(np.arcsin((init_r / 2.0) / concentric_r)))))
        theta = np.linspace(0.0, 360.0, n_circ + 1)[:-1]
        x = concentric_r * np.cos(np.radians(theta))
        y = concentric_r * np.sin(np.radians(theta))
        cntr  = np.vstack([cntr,  np.column_stack([x, y])])
        radii = np.concatenate([radii, init_r * np.ones(n_circ)])
    # concentric_r is now 19 * 2 * 60 = 2280 at loop exit

    # ------------------------------------------------------------------
    # Override: unfiltered foveola (binary closing to fill gaps)
    # ------------------------------------------------------------------
    log("Applying unfiltered foveola override ...")
    half_sz = 479
    sz = 2 * half_sz + 1
    tmp_mask = np.zeros((sz, sz), dtype=np.uint8)

    # Convert arc-second coords -> 0-based pixel indices (row=y, col=x)
    rows_f = np.clip(np.round(cntr[:, 1] / spp + half_sz).astype(int), 0, sz - 1)
    cols_f = np.clip(np.round(cntr[:, 0] / spp + half_sz).astype(int), 0, sz - 1)
    tmp_mask[rows_f, cols_f] = 1

    # bwmorph(mask, 'close') = morphological closing with 3×3 ones
    tmp_mask = binary_closing(tmp_mask, structure=np.ones((3, 3))).astype(np.uint8)

    rows_nz, cols_nz = np.nonzero(tmp_mask)
    # MATLAB: spp * ([r, c] - 1 - half_sz) with 1-based r,c  ≡  spp * (0based - half_sz)
    cntr  = spp * np.column_stack([cols_nz - half_sz, rows_nz - half_sz]).astype(float)
    radii = init_r * np.ones(len(rows_nz))

    # ------------------------------------------------------------------
    # Variable-resolution: fovea + periphery (RF grows with eccentricity)
    # ------------------------------------------------------------------
    log("Building fovea/periphery cell layout ...")
    odd_fl = True
    while concentric_r <= (model['fov_in_degrees'] / 2.0) * 3600.0:
        radius   = init_r + model['var_ecc_rf_factor'] * concentric_r
        n_circ   = int(np.ceil(360.0 / (2.0 * np.degrees(np.arcsin((radius / 2.0) / concentric_r)))))
        theta    = np.linspace(0.0, 360.0, n_circ + 1)[:-1]
        if odd_fl:
            odd_fl = False
        else:
            theta  = theta + round(360.0 / n_circ / 2.0)
            odd_fl = True
        x = concentric_r * np.cos(np.radians(theta))
        y = concentric_r * np.sin(np.radians(theta))
        cntr  = np.vstack([cntr,  np.column_stack([x, y])])
        radii = np.concatenate([radii, radius * np.ones(n_circ)])
        concentric_r += radius

    var_cntr  = cntr.copy()
    var_radii = radii.copy()

    # ------------------------------------------------------------------
    # Gaussian filters for variable model
    # ------------------------------------------------------------------
    log(f"Computing Gaussian filters for variable model ({len(var_radii)} cells) ...")
    var_roi_half, var_tar_x, var_tar_y, var_src_x, var_src_y = \
        _build_xlim_ylim(var_cntr, var_radii, spp, gauss_half)
    var_filter = _build_gaussian_filters(var_cntr, var_radii, spp, gauss_half)

    # ------------------------------------------------------------------
    # Constant-resolution model (uniform spacing)
    # ------------------------------------------------------------------
    log("Building constant resolution cell layout ...")
    c_r     = model['const_rf_radius']
    n_half  = int(round(model['const_half_num_of_uniform_rf_cells']))

    c_cntr  = np.array([[0.0, 0.0]])
    c_radii = np.array([c_r])
    odd_fl  = True

    for nn in range(2, n_half + 1):
        concentric_r = (nn - 1) * c_r
        n_circ  = int(np.ceil(360.0 / (2.0 * np.degrees(np.arcsin((c_r / 2.0) / concentric_r)))))
        theta   = np.linspace(0.0, 360.0, n_circ + 1)[:-1]
        if odd_fl:
            odd_fl = False
        else:
            theta  = theta + round(360.0 / n_circ / 2.0)
            odd_fl = True
        x = concentric_r * np.cos(np.radians(theta))
        y = concentric_r * np.sin(np.radians(theta))
        c_cntr  = np.vstack([c_cntr,  np.column_stack([x, y])])
        c_radii = np.concatenate([c_radii, c_r * np.ones(n_circ)])

    # ------------------------------------------------------------------
    # Gaussian filters for constant model
    # ------------------------------------------------------------------
    log(f"Computing Gaussian filters for constant model ({len(c_radii)} cells) ...")
    const_roi_half, const_tar_x, const_tar_y, const_src_x, const_src_y = \
        _build_xlim_ylim(c_cntr, c_radii, spp, gauss_half)
    const_filter = _build_gaussian_filters(c_cntr, c_radii, spp, gauss_half)

    # ------------------------------------------------------------------
    # Save to .mat (HDF5 v7.3) — filename expected by filter_image.py
    # ------------------------------------------------------------------
    if save_to_file:
        if output_dir is None:
            output_dir = os.path.dirname(os.path.realpath(__file__))
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'sampling_scheme_params.mat')
        log(f"Saving to {out_path} ...")
        data = {
            # scalar shared
            'seconds_per_pixel': np.array([[spp]]),
            # variable model
            'var_foveola_cell_radius':      np.array([[model['var_foveola_cell_radius']]]),
            'var_rf_roi_half_sz':           np.array([[var_roi_half]]),
            'var_rf_circles_cntr_xy':       var_cntr.astype(np.float64),
            'var_rf_circles_radii':         var_radii.reshape(-1, 1).astype(np.float64),
            'var_rf_filter':                var_filter,
            'var_max_out_img_sz':           np.array([[model['var_max_out_img_sz']]]),
            'var_tar_xlim':                 var_tar_x,
            'var_tar_ylim':                 var_tar_y,
            'var_src_xlim':                 var_src_x,
            'var_src_ylim':                 var_src_y,
            # constant model
            'const_rf_radius':              np.array([[model['const_rf_radius']]]),
            'const_rf_roi_half_sz':         np.array([[const_roi_half]]),
            'const_rf_circles_cntr_xy':     c_cntr.astype(np.float64),
            'const_rf_circles_radii':       c_radii.reshape(-1, 1).astype(np.float64),
            'const_rf_filter':              const_filter,
            'const_half_num_of_uniform_rf_cells': np.array([[model['const_half_num_of_uniform_rf_cells']]]),
            'const_max_out_img_sz':         np.array([[model['const_max_out_img_sz']]]),
            'const_tar_xlim':               const_tar_x,
            'const_tar_ylim':               const_tar_y,
            'const_src_xlim':               const_src_x,
            'const_src_ylim':               const_src_y,
        }
        hdf5storage.savemat(out_path, data)
        log(f"Done! Saved to {out_path}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate foveated sampling maps (Python port of MATLAB scripts)."
    )
    parser.add_argument(
        '--model_index', type=int, default=1,
        help='Pixel budget: 1..11 -> [3%%, 10%%, 15%%, 20%%, 30%%, 40%%, 50%%, 60%%, 70%%, 80%%, 90%%]'
    )
    parser.add_argument(
        '--fov_index', type=int, default=2,
        help='Field of view: 1->27°, 2->30°, 3->54°'
    )
    parser.add_argument(
        '--img_height', type=int, default=480,
        help='Image height in pixels (default: 480)'
    )
    parser.add_argument(
        '--img_width', type=int, default=640,
        help='Image width in pixels (default: 640)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save sampling_scheme_params.mat (default: same folder as this script)'
    )
    args = parser.parse_args()

    if args.model_index not in range(1, 12):
        parser.error('--model_index must be between 1 and 11')
    if args.fov_index not in [1, 2, 3]:
        parser.error('--fov_index must be 1, 2, or 3')

    generic_inverted_pyramid_model(
        model_index=args.model_index,
        fov_index=args.fov_index,
        img_size=(args.img_height, args.img_width),
        save_to_file=True,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
