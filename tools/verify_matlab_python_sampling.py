#!/usr/bin/env python3
"""Verify MATLAB and Python sampling-map generation end to end.

The script generates sampling .mat files with both implementations, compares
the variables used by sampling_schemes/filter_image.py, filters a small image
set with each .mat file, and compares the decoded output pixels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hdf5storage
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_DIR = REPO_ROOT / "matlab"
SAMPLING_DIR = REPO_ROOT / "sampling_schemes"
SAMPLE_PERCENTAGES = [3, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
FOV_DEGREES = [27, 30, 54]

MAT_KEYS = [
    "seconds_per_pixel",
    "var_foveola_cell_radius",
    "var_rf_roi_half_sz",
    "var_rf_circles_cntr_xy",
    "var_rf_circles_radii",
    "var_rf_filter",
    "var_max_out_img_sz",
    "var_tar_xlim",
    "var_tar_ylim",
    "var_src_xlim",
    "var_src_ylim",
    "const_rf_radius",
    "const_rf_roi_half_sz",
    "const_rf_circles_cntr_xy",
    "const_rf_circles_radii",
    "const_rf_filter",
    "const_half_num_of_uniform_rf_cells",
    "const_max_out_img_sz",
    "const_tar_xlim",
    "const_tar_ylim",
    "const_src_xlim",
    "const_src_ylim",
]

FILTER_KEYS_BY_TYPE = {
    "const": [
        "seconds_per_pixel",
        "const_rf_circles_cntr_xy",
        "const_rf_filter",
        "const_rf_roi_half_sz",
        "const_max_out_img_sz",
        "const_tar_xlim",
        "const_tar_ylim",
        "const_src_xlim",
        "const_src_ylim",
    ],
    "var": [
        "seconds_per_pixel",
        "var_rf_circles_cntr_xy",
        "var_rf_filter",
        "var_rf_roi_half_sz",
        "var_max_out_img_sz",
        "var_tar_xlim",
        "var_tar_ylim",
        "var_src_xlim",
        "var_src_ylim",
    ],
}


@dataclass
class ArrayComparison:
    key: str
    matlab_shape: tuple[int, ...] | None
    python_shape: tuple[int, ...] | None
    same_shape: bool
    exact_equal: bool
    max_abs_diff: float | None
    mean_abs_diff: float | None
    differing_values: int | None
    first_difference: list[int] | None


@dataclass
class ImageComparison:
    sampling_type: str
    image_name: str
    matlab_image: str
    python_image: str
    same_shape: bool
    exact_equal: bool
    differing_pixels: int | None
    differing_values: int | None
    max_abs_diff: int | None
    mean_abs_diff: float | None


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_bash(script: str, cwd: Path | None = None) -> None:
    print("+ bash -lc", script, flush=True)
    subprocess.run(["bash", "-lc", script], cwd=str(cwd) if cwd else None, check=True)


def matlab_literal(path: Path) -> str:
    return str(path).replace("'", "''")


def generate_matlab_map(
    out_dir: Path,
    model_index: int,
    fov_index: int,
    img_height: int,
    img_width: int,
    matlab_module: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    percent = SAMPLE_PERCENTAGES[model_index - 1]
    fov = FOV_DEGREES[fov_index - 1]
    output = out_dir / f"sampling_models_{percent}_samp_per_{fov}_fov_deg_unfilt_foveola.mat"
    batch = (
        f"addpath('{matlab_literal(MATLAB_DIR)}'); "
        f"cd('{matlab_literal(out_dir)}'); "
        f"generic_inverted_pyramid_model({model_index}, {fov_index}, "
        f"[{img_height}, {img_width}], true);"
    )
    run_bash(f"module load {matlab_module}; matlab -batch \"{batch}\"")
    if not output.exists():
        raise FileNotFoundError(f"MATLAB did not create expected file: {output}")
    return output


def generate_python_map(
    out_dir: Path,
    model_index: int,
    fov_index: int,
    img_height: int,
    img_width: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "sampling_scheme_params.mat"
    run(
        [
            sys.executable,
            str(MATLAB_DIR / "generate_sampling_maps.py"),
            "--model_index",
            str(model_index),
            "--fov_index",
            str(fov_index),
            "--img_height",
            str(img_height),
            "--img_width",
            str(img_width),
            "--output_dir",
            str(out_dir),
        ]
    )
    if not output.exists():
        raise FileNotFoundError(f"Python did not create expected file: {output}")
    return output


def load_mat(path: Path, keys: list[str]) -> dict[str, Any]:
    return hdf5storage.loadmat(str(path), variable_names=keys)


def first_difference_index(diff_mask: np.ndarray) -> list[int] | None:
    if not np.any(diff_mask):
        return None
    return [int(v) for v in np.argwhere(diff_mask)[0]]


def compare_mat_files(matlab_mat: Path, python_mat: Path, atol: float, rtol: float) -> list[ArrayComparison]:
    matlab_data = load_mat(matlab_mat, MAT_KEYS)
    python_data = load_mat(python_mat, MAT_KEYS)
    comparisons: list[ArrayComparison] = []

    for key in MAT_KEYS:
        m_val = matlab_data.get(key)
        p_val = python_data.get(key)
        if m_val is None or p_val is None:
            comparisons.append(
                ArrayComparison(
                    key=key,
                    matlab_shape=None if m_val is None else tuple(np.asarray(m_val).shape),
                    python_shape=None if p_val is None else tuple(np.asarray(p_val).shape),
                    same_shape=False,
                    exact_equal=False,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    differing_values=None,
                    first_difference=None,
                )
            )
            continue

        m_arr = np.asarray(m_val)
        p_arr = np.asarray(p_val)
        same_shape = m_arr.shape == p_arr.shape
        exact_equal = same_shape and np.array_equal(m_arr, p_arr)
        max_abs_diff = None
        mean_abs_diff = None
        differing_values = None
        first_diff = None

        if same_shape and np.issubdtype(m_arr.dtype, np.number) and np.issubdtype(p_arr.dtype, np.number):
            diff = np.abs(m_arr.astype(np.float64) - p_arr.astype(np.float64))
            max_abs_diff = float(np.nanmax(diff)) if diff.size else 0.0
            mean_abs_diff = float(np.nanmean(diff)) if diff.size else 0.0
            diff_mask = ~np.isclose(m_arr, p_arr, atol=atol, rtol=rtol, equal_nan=True)
            differing_values = int(np.count_nonzero(diff_mask))
            first_diff = first_difference_index(diff_mask)

        comparisons.append(
            ArrayComparison(
                key=key,
                matlab_shape=tuple(int(v) for v in m_arr.shape),
                python_shape=tuple(int(v) for v in p_arr.shape),
                same_shape=same_shape,
                exact_equal=exact_equal,
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
                differing_values=differing_values,
                first_difference=first_diff,
            )
        )

    return comparisons


def configure_filter_module(mat_path: Path, image_root: Path, out_root: Path, sampling_type: str) -> None:
    if str(SAMPLING_DIR) not in sys.path:
        sys.path.insert(0, str(SAMPLING_DIR))
    import filter_image as filter_mod  # type: ignore

    prefix_idx = {"const": 0, "var": 1}[sampling_type]
    mat_prefix = sampling_type
    contents = load_mat(mat_path, FILTER_KEYS_BY_TYPE[sampling_type])
    seconds_per_pixel = contents["seconds_per_pixel"][0][0]
    centers = contents[f"{mat_prefix}_rf_circles_cntr_xy"]

    filter_mod.bypass_filter = False
    filter_mod.processing_buffer = 1
    filter_mod.old_database_path = str(image_root)
    filter_mod.new_database_path = str(out_root)
    filter_mod.is_full_fov = False
    filter_mod.roi_half_sz = contents[f"{mat_prefix}_rf_roi_half_sz"][0][0].astype(int)
    filter_mod.calc_pts = filter_mod.roi_half_sz + np.floor(centers / seconds_per_pixel + 0.5)[:, ::-1].astype(int)
    filter_mod.ch1_mode = True if prefix_idx == 0 else False
    filter_mod.filter = contents[f"{mat_prefix}_rf_filter"]
    filter_mod.max_out_img_sz = contents[f"{mat_prefix}_max_out_img_sz"][0][0].astype(int)
    filter_mod.tar_xlim = contents[f"{mat_prefix}_tar_xlim"].astype(int)
    filter_mod.tar_ylim = contents[f"{mat_prefix}_tar_ylim"].astype(int)
    filter_mod.src_xlim = contents[f"{mat_prefix}_src_xlim"].astype(int)
    filter_mod.src_ylim = contents[f"{mat_prefix}_src_ylim"].astype(int)


def filter_one_image(
    mat_path: Path,
    image_root: Path,
    image_name: str,
    out_root: Path,
    sampling_type: str,
    fp_shift_x: int,
    fp_shift_y: int,
) -> Path:
    if str(SAMPLING_DIR) not in sys.path:
        sys.path.insert(0, str(SAMPLING_DIR))
    import filter_image as filter_mod  # type: ignore

    out_root.mkdir(parents=True, exist_ok=True)
    configure_filter_module(mat_path, image_root, out_root, sampling_type)

    with Image.open(image_root / image_name) as img:
        fixation = [
            min(max((img.size[0] // 2) + fp_shift_x, 0), img.size[0]),
            min(max((img.size[1] // 2) + fp_shift_y, 0), img.size[1]),
        ]

    out_name = "%s_oid_%d_fpx_%d_fpy_%d.jpg" % (
        str(Path(image_name).with_suffix("")),
        999,
        fixation[0],
        fixation[1],
    )
    out_path = out_root / out_name
    if out_path.exists():
        out_path.unlink()
    filter_mod.filter_preprocessing(image_name, out_name, fixation)
    if not out_path.exists():
        raise FileNotFoundError(f"Filtering did not create expected file: {out_path}")
    return out_path


def compare_images(
    sampling_type: str,
    image_name: str,
    matlab_image: Path,
    python_image: Path,
) -> ImageComparison:
    m_arr = np.asarray(Image.open(matlab_image).convert("RGB"))
    p_arr = np.asarray(Image.open(python_image).convert("RGB"))
    same_shape = m_arr.shape == p_arr.shape
    if not same_shape:
        return ImageComparison(
            sampling_type=sampling_type,
            image_name=image_name,
            matlab_image=str(matlab_image),
            python_image=str(python_image),
            same_shape=False,
            exact_equal=False,
            differing_pixels=None,
            differing_values=None,
            max_abs_diff=None,
            mean_abs_diff=None,
        )

    diff = np.abs(m_arr.astype(np.int16) - p_arr.astype(np.int16))
    value_mask = diff != 0
    pixel_mask = np.any(value_mask, axis=2)
    return ImageComparison(
        sampling_type=sampling_type,
        image_name=image_name,
        matlab_image=str(matlab_image),
        python_image=str(python_image),
        same_shape=True,
        exact_equal=bool(not np.any(value_mask)),
        differing_pixels=int(np.count_nonzero(pixel_mask)),
        differing_values=int(np.count_nonzero(value_mask)),
        max_abs_diff=int(np.max(diff)) if diff.size else 0,
        mean_abs_diff=float(np.mean(diff)) if diff.size else 0.0,
    )


def select_images(image_root: Path, limit: int) -> list[str]:
    suffixes = {".jpg", ".jpeg", ".png"}
    images = sorted(p.name for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    if not images:
        raise FileNotFoundError(f"No images found in {image_root}")
    return images[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_index", type=int, default=1, help="1..11, matching MATLAB indexing")
    parser.add_argument("--fov_index", type=int, default=2, help="1..3, matching MATLAB indexing")
    parser.add_argument("--img_height", type=int, default=480)
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--image_root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--image_limit", type=int, default=1)
    parser.add_argument("--sampling_type", choices=["const", "var", "both"], default="both")
    parser.add_argument("--fp_shift_x", type=int, default=0)
    parser.add_argument("--fp_shift_y", type=int, default=0)
    parser.add_argument("--matlab_module", default="matlab/matlab-2025b")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "verification_outputs")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-7)
    args = parser.parse_args()

    if args.model_index not in range(1, 12):
        parser.error("--model_index must be 1..11")
    if args.fov_index not in [1, 2, 3]:
        parser.error("--fov_index must be 1, 2, or 3")

    run_id = time.strftime("verify_%Y%m%d_%H%M%S")
    root = args.out_dir / run_id
    matlab_dir = root / "matlab"
    python_dir = root / "python"
    filtered_dir = root / "filtered"
    root.mkdir(parents=True, exist_ok=True)

    print(f"Writing verification outputs to {root}", flush=True)
    matlab_mat = generate_matlab_map(
        matlab_dir,
        args.model_index,
        args.fov_index,
        args.img_height,
        args.img_width,
        args.matlab_module,
    )
    python_mat = generate_python_map(
        python_dir,
        args.model_index,
        args.fov_index,
        args.img_height,
        args.img_width,
    )
    shutil.copy2(matlab_mat, matlab_dir / "sampling_scheme_params.mat")

    mat_comparisons = compare_mat_files(matlab_mat, python_mat, args.atol, args.rtol)
    image_names = select_images(args.image_root, args.image_limit)
    sampling_types = ["const", "var"] if args.sampling_type == "both" else [args.sampling_type]
    image_comparisons: list[ImageComparison] = []

    for sampling_type in sampling_types:
        for image_name in image_names:
            matlab_img = filter_one_image(
                matlab_mat,
                args.image_root,
                image_name,
                filtered_dir / "matlab" / sampling_type,
                sampling_type,
                args.fp_shift_x,
                args.fp_shift_y,
            )
            python_img = filter_one_image(
                python_mat,
                args.image_root,
                image_name,
                filtered_dir / "python" / sampling_type,
                sampling_type,
                args.fp_shift_x,
                args.fp_shift_y,
            )
            image_comparisons.append(compare_images(sampling_type, image_name, matlab_img, python_img))

    report = {
        "config": {
            "model_index": args.model_index,
            "fov_index": args.fov_index,
            "img_height": args.img_height,
            "img_width": args.img_width,
            "image_root": str(args.image_root),
            "image_limit": args.image_limit,
            "sampling_type": args.sampling_type,
            "fp_shift_x": args.fp_shift_x,
            "fp_shift_y": args.fp_shift_y,
            "matlab_module": args.matlab_module,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "paths": {
            "root": str(root),
            "matlab_mat": str(matlab_mat),
            "python_mat": str(python_mat),
        },
        "mat_comparisons": [asdict(c) for c in mat_comparisons],
        "image_comparisons": [asdict(c) for c in image_comparisons],
    }

    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    bad_mat = [c for c in mat_comparisons if not c.same_shape or (c.differing_values not in (None, 0))]
    bad_images = [c for c in image_comparisons if not c.exact_equal]

    print(f"\nReport: {report_path}")
    print(f"MAT variables compared: {len(mat_comparisons)}")
    print(f"MAT variables with shape/value differences beyond tolerance: {len(bad_mat)}")
    for c in bad_mat[:10]:
        print(
            f"  {c.key}: shape {c.matlab_shape} vs {c.python_shape}, "
            f"diff_values={c.differing_values}, max_abs_diff={c.max_abs_diff}, "
            f"first_difference={c.first_difference}"
        )

    print(f"Images compared: {len(image_comparisons)}")
    print(f"Images with pixel differences: {len(bad_images)}")
    for c in bad_images:
        print(
            f"  {c.sampling_type}/{c.image_name}: differing_pixels={c.differing_pixels}, "
            f"differing_values={c.differing_values}, max_abs_diff={c.max_abs_diff}, "
            f"mean_abs_diff={c.mean_abs_diff}"
        )

    return 1 if bad_mat or bad_images else 0


if __name__ == "__main__":
    raise SystemExit(main())
