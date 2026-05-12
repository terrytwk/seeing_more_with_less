import cv2 as cv
import numpy as np
import utils
import math
import torch
from opt_einsum import contract
import os
from pathlib import Path


def dataset_name_from_path(image_root):
    image_root = Path(image_root)
    if image_root.parent.name == "coco" and image_root.name in {"train2017", "val2017", "test2017"}:
        return f"coco_{image_root.name}"
    return image_root.name
import scipy.interpolate
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import operator
import json
from datetime import datetime

def relu(x): # Regular ReLU function
    return max(0, x)

def drelu(x, m): # Double ended ReLU function, a linear function between for values between 0 and m, otherwise 0 and m at those values above or below respectively
    return min(relu(x), m)

def configure_filter_globals(config):
    global bypass_filter; bypass_filter = False
    global processing_buffer; processing_buffer = config["batchsize"]
    global old_database_path; old_database_path = config["path"]
    global new_database_path; new_database_path = config["new_database_path"]
    global is_full_fov; is_full_fov = False
    global filter_run_type; filter_run_type = config["type"]
    global filter_mat_prefix; filter_mat_prefix = config["mat_prefix"]
    global filter_prefix_idx; filter_prefix_idx = config["prefix_idx"]
    global filter_fixation_json_root; filter_fixation_json_root = config["fixation_json_root"]
    global filter_fixation_json_only; filter_fixation_json_only = config["fixation_json_only"]
    global filter_fp_shift_x; filter_fp_shift_x = config["fp_shift_x"]
    global filter_fp_shift_y; filter_fp_shift_y = config["fp_shift_y"]


def load_filter_parameters():
    if 'seconds_per_pixel' in globals():
        return

    print("Reading in MAT file", flush=True)
    mat_contents = utils.read_mat('sampling_scheme_params.mat', [
        'seconds_per_pixel',
        filter_mat_prefix + '_rf_circles_cntr_xy',
        filter_mat_prefix + '_rf_filter',
        filter_mat_prefix + '_rf_roi_half_sz',
        filter_mat_prefix + '_max_out_img_sz',
        filter_mat_prefix + '_tar_xlim',
        filter_mat_prefix + '_tar_ylim',
        filter_mat_prefix + '_src_xlim',
        filter_mat_prefix + '_src_ylim',
    ])
    print("Finished reading MAT file", flush=True)

    global seconds_per_pixel; seconds_per_pixel = mat_contents['seconds_per_pixel'][0][0]
    circles_cntr_xy = mat_contents[filter_mat_prefix + '_rf_circles_cntr_xy']

    global roi_half_sz; roi_half_sz = mat_contents[filter_mat_prefix + '_rf_roi_half_sz'][0][0].astype(int)
    global calc_pts; calc_pts = roi_half_sz + np.floor(circles_cntr_xy / seconds_per_pixel + 0.5)[:, ::-1].astype(int)
    global ch1_mode; ch1_mode = True if filter_prefix_idx == 0 else False
    global filter; filter = mat_contents[filter_mat_prefix + '_rf_filter']
    global max_out_img_sz; max_out_img_sz = mat_contents[filter_mat_prefix + '_max_out_img_sz'][0][0].astype(int)
    global tar_xlim; tar_xlim = mat_contents[filter_mat_prefix + '_tar_xlim'].astype(int)
    global tar_ylim; tar_ylim = mat_contents[filter_mat_prefix + '_tar_ylim'].astype(int)
    global src_xlim; src_xlim = mat_contents[filter_mat_prefix + '_src_xlim'].astype(int)
    global src_ylim; src_ylim = mat_contents[filter_mat_prefix + '_src_ylim'].astype(int)
    mat_contents.clear()


def get_fixation_points(image_name):
    fixation_points = [[0, 0]]
    obj_ids = [999]
    if filter_run_type == 'const':
        return fixation_points, obj_ids

    fixation_json_root = filter_fixation_json_root
    if fixation_json_root is None:
        fixation_json_root = os.path.join("outputs", "fixations", dataset_name_from_path(old_database_path), "default")
    json_info_filename = os.path.join(fixation_json_root, str(Path(image_name).with_suffix('.json')))
    if os.path.exists(json_info_filename):
        with open(json_info_filename) as fp_json_file:
            json_data = json.load(fp_json_file)
        objects_info = json_data["objects_info"]
    else:
        objects_info = []

    if objects_info and filter_fixation_json_only:
        fixation_points = []
        obj_ids = []

    if objects_info:
        for obj_info in objects_info:
            fixation_points.append([obj_info['centroid'][1],obj_info['centroid'][0]])
            obj_ids.append(obj_info['obj_id'])

    return fixation_points, obj_ids


def filter_image(img):

    # print('init filters: ' + str(datetime.now()))
    out_img = np.full((img.shape[0], img.shape[1], img.shape[2]), 128)
    num_of_sets = int(filter.shape[2]/1500)
    sets = np.array_split(range(filter.shape[2]),num_of_sets)
    tmp_filter = np.full((img.shape[0], img.shape[1], len(sets[0])), 0, dtype=float)
    # print('start filtering process: ' + str(datetime.now()))
    for s in range(len(sets)):
        for i in range(len(sets[s])):
            tmp_filter[tar_ylim[sets[s][i], 0]-1:tar_ylim[sets[s][i], 1],tar_xlim[sets[s][i], 0]-1:tar_xlim[sets[s][i], 1],i] = filter[src_ylim[sets[s][i], 0]-1:src_ylim[sets[s][i], 1], src_xlim[sets[s][i], 0]-1:src_xlim[sets[s][i], 1],sets[s][i]]
         #tmp_filter[tar_ylim[si, 0]:tar_ylim[si, 1], tar_xlim[si, 0]:tar_xlim[si, 1], :] = np.repeat(filter[src_ylim[si, 0]:src_ylim[si, 1], src_xlim[si, 0]:src_xlim[si, 1], si][:, :, np.newaxis], 3, axis=2)
         #out_img[calc_pts[si, 0], calc_pts[si, 1]] = np.dstack(np.floor(contract("ijk,ijq->qk", torch.from_numpy(tmp_filter).float(), torch.from_numpy(img).float(),backend='torch') + 0.5)).squeeze()
        # print('apply filter: ' + str(datetime.now()))
        # out_img[calc_pts[sets[s], 0], calc_pts[sets[s], 1]] = np.dstack(np.floor(contract("ijk,ijq->qk", torch.from_numpy(tmp_filter).float(), torch.from_numpy(img).float(),backend='torch') + 0.5)).squeeze()
        out_img[calc_pts[sets[s], 0], calc_pts[sets[s], 1]] = np.dstack(np.floor(np.einsum("ijk,ijq->qk", tmp_filter, img) + 0.5)).squeeze()
        if s<(len(sets)-1):
            if len(sets[s+1])==len(sets[s]):
                for i in range(len(sets[s])):
                    tmp_filter[tar_ylim[sets[s][i], 0]-1:tar_ylim[sets[s][i], 1], tar_xlim[sets[s][i], 0]-1:tar_xlim[sets[s][i], 1], i] = 0
            else:
                tmp_filter = np.full((img.shape[0], img.shape[1], len(sets[s+1])), 0, dtype=float)

    # print('apply interpolation: ' + str(datetime.now()))
    var_indn = np.full((out_img.shape[0], out_img.shape[1]), True)
    var_indn[calc_pts[:, 0], calc_pts[:, 1]] = False
    new_var_y_x = np.transpose(np.nonzero(var_indn))
    for channel in range(3):
        src_vals = out_img[:, :, channel][calc_pts[:, 0], calc_pts[:, 1]]
        interp = scipy.interpolate.griddata(calc_pts, src_vals, new_var_y_x, method='linear')
        nan_mask = np.isnan(interp)
        if nan_mask.any():
            interp[nan_mask] = scipy.interpolate.griddata(
                calc_pts, src_vals, new_var_y_x[nan_mask], method='nearest')
        out_img[:, :, channel][var_indn] = interp
    out_img[np.where(out_img < 0)] = 0
    out_img[np.where(out_img > 255)] = 255
    # print('Done: ' + str(datetime.now()))
    return out_img

def filter_preprocessing(org_image_name, out_image_name, fixation_point):
    pil_img = Image.open(os.path.join(old_database_path, org_image_name))
#    pil_img = Image.open(os.path.join(old_database_path,'try',org_image_name))
    if len(pil_img.getbands()) == 1:
        tmp_img = Image.new('RGB',pil_img.size)
        tmp_img.paste(pil_img)
        pil_img = tmp_img.copy()

    pre_pad_sz = [1, 1, 0]
    post_pad_sz = [1, 1, 0]
    if (fixation_point[0] < roi_half_sz) | ((fixation_point[0] + roi_half_sz) > pil_img.size[0]):
        # x dimension
        pre_pad_sz[0] = np.maximum(0, roi_half_sz - fixation_point[0] + 1)
        post_pad_sz[0] = np.maximum(0, roi_half_sz + fixation_point[0] - pil_img.size[0])
        new_img_size = tuple(map(operator.add, pil_img.size ,(pre_pad_sz[0] + post_pad_sz[0],0)))
        in_img = Image.new("RGB", new_img_size, color=(128,128,128))
        in_img.paste(pil_img,(pre_pad_sz[0],0))
    else:
        in_img = pil_img.crop((fixation_point[0] - roi_half_sz, 0, fixation_point[0]+roi_half_sz+1, pil_img.size[0]))
        pre_pad_sz[0] = roi_half_sz - max_out_img_sz // 2
        post_pad_sz[0] = roi_half_sz - max_out_img_sz // 2

    if (fixation_point[1] < roi_half_sz) | ((fixation_point[1] + roi_half_sz) > pil_img.size[1]):
        # y dimension
        pre_pad_sz[1] = np.maximum(0, roi_half_sz - fixation_point[1] + 1)
        post_pad_sz[1] = np.maximum(0, roi_half_sz + fixation_point[1] - pil_img.size[1])
        new_img_size = tuple(map(operator.add, in_img.size ,(0,pre_pad_sz[1] + post_pad_sz[1])))
        tmp_img = in_img
        in_img = Image.new("RGB", new_img_size, color=(128,128,128))
        in_img.paste(tmp_img, (0, pre_pad_sz[1]))
    else:
        in_img = in_img.crop((0, fixation_point[1] - roi_half_sz, in_img.size[1], fixation_point[1] + roi_half_sz + 1))
        pre_pad_sz[1] = roi_half_sz - max_out_img_sz // 2
        post_pad_sz[1] = roi_half_sz - max_out_img_sz // 2

    in_img = in_img.crop((0, 0, roi_half_sz * 2 + 1, roi_half_sz * 2 + 1))
    out_img_crop = (np.maximum(pre_pad_sz[0], roi_half_sz - max_out_img_sz // 2 + 1),
                    np.maximum(pre_pad_sz[1], roi_half_sz - max_out_img_sz // 2 + 1),
                    np.maximum(pre_pad_sz[0], roi_half_sz - max_out_img_sz // 2 + 1) + np.minimum(pil_img.size[0], max_out_img_sz),
                    np.maximum(pre_pad_sz[1], roi_half_sz - max_out_img_sz // 2 + 1) + np.minimum(pil_img.size[1], max_out_img_sz))


    # start = time.time()
    if not bypass_filter:
        filtered_image = filter_image(np.array(in_img))
    else:
        filtered_image = in_img

    out_img = Image.fromarray(np.uint8(filtered_image))
    out_img = out_img.crop(out_img_crop)
    # print("Application of filter complete. Total elapsed time: " + str(time.time() - start) + " seconds")
    if is_full_fov:
        out_tmp_img = Image.new('RGB', pil_img.size, 0)
#        crop_bbox = [ fixation_point[0]-(out_img.size[0]//2), fixation_point[1]-(out_img.size[1]//2), fixation_point[0]-(out_img.size[0]//2)+out_img.size[0], fixation_point[1]-(out_img.size[1]//2)+out_img.size[1] ]
        crop_bbox = [fixation_point[0] - (out_img.size[0] // 2), fixation_point[1] - (out_img.size[1] // 2),
                     fixation_point[0] - (out_img.size[0] // 2) + out_img.size[0],
                     fixation_point[1] - (out_img.size[1] // 2) + out_img.size[1]]
        out_tmp_img.paste(out_img, crop_bbox)
        out_img = out_tmp_img

    out_img.save(os.path.join(new_database_path, out_image_name),format='jpeg',quality=96,subsampling=0)
    img_bbox = (np.maximum(0,fixation_point[0]-max_out_img_sz//2),
                np.maximum(0,fixation_point[1]-max_out_img_sz//2),
                np.minimum(pil_img.size[0],fixation_point[0]+max_out_img_sz//2),
                np.minimum(pil_img.size[1],fixation_point[1]+max_out_img_sz//2))
    return out_img, img_bbox


def process_image(image_name):
    now = datetime.now()
    print(now.strftime("%d/%m/%Y %H:%M:%S") + " Processing image: " + image_name + " in " + new_database_path, flush=True)
    generated = 0
    skipped = 0
    errors = 0

    fixation_points, obj_ids = get_fixation_points(image_name)
    for ind in range(len(fixation_points)):
        try:
            with Image.open(os.path.join(old_database_path, image_name)) as pil_img:
                image_size = pil_img.size
            if fixation_points[ind][0] == 0:
                fixation_points[ind] = [drelu((image_size[0] // 2) + filter_fp_shift_x, image_size[0]), drelu((image_size[1] // 2) + filter_fp_shift_y, image_size[1])]

            new_image_name = '%s_oid_%d_fpx_%d_fpy_%d.jpg' % (str(Path(image_name).with_suffix('')), obj_ids[ind], fixation_points[ind][0], fixation_points[ind][1])
            if os.path.exists(os.path.join(new_database_path, new_image_name)):
                skipped += 1
                continue

            try:
                load_filter_parameters()
                filter_preprocessing(image_name, new_image_name, fixation_points[ind])
                generated += 1
            except Exception as exc:
                errors += 1
                print("There are some errors generating image: " + new_image_name + ". skipping image... " + str(exc), flush=True)
        except Exception as exc:
            errors += 1
            print("There are some errors loading image: " + image_name + ". skipping it... " + str(exc), flush=True)

    return {"image": image_name, "generated": generated, "skipped": skipped, "errors": errors}


def run_images(images, num_pool_threads, config=None):
    totals = {"generated": 0, "skipped": 0, "errors": 0}
    if num_pool_threads <= 1:
        for image_name in images:
            result = process_image(image_name)
            for key in totals:
                totals[key] += result[key]
        return totals

    print("Processing " + str(len(images)) + " images with " + str(num_pool_threads) + " workers", flush=True)
    with ProcessPoolExecutor(max_workers=num_pool_threads, initializer=configure_filter_globals, initargs=(config,)) as executor:
        futures = [executor.submit(process_image, image_name) for image_name in images]
        for future in as_completed(futures):
            try:
                result = future.result()
                for key in totals:
                    totals[key] += result[key]
            except Exception as exc:
                totals["errors"] += 1
                print("Worker failed while processing an image. skipping it... " + str(exc), flush=True)
    return totals

def main():
    parser = argparse.ArgumentParser(description="Images Generator")
    parser.add_argument(
        "--path",
        default="",
        help="Full path to source images at full resolution",
    )
    parser.add_argument(
        "--outfolder",
        default="out",
        help="Output path prefix. The script appends _<fov>d_<budget>perc/{Constant,Variable}.",
    )
    parser.add_argument(
        "--fov_index",
        default=0,
        type=int,
        help="Field of view index. Options: 0 -> 27deg, 1 -> 30deg, 2 -> 54deg",
    )
    parser.add_argument(
        "--model_index",
        default=0,
        type=int,
        help="Models of different sample percentage. Options are: 0-10 -> [3%%, 10%%, 15%%, 20%%, 30%%, ... 90%%]",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="Range of images to apply filters to (in hundreds)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=50,
        help="Number of images to process",
    )
    parser.add_argument(
        "--type",
        default="None",
        help="Type of filter to apply image to",
    )
    parser.add_argument(
        "--pool_threads",
        default=0,
        type=int,
        help="Number of pool threads to use (ie number of concurrent images to process",
    )
    parser.add_argument(
        "--fp_shift_x",
        default=0,
        type=int,
        help="Number of pixels to shift the fixation point in the horizontal direction from the image center",
    )
    parser.add_argument(
        "--fp_shift_y",
        default=0,
        type=int,
        help="Number of pixels to shift the fixation point in the vertical direction from the image center",
    )
    parser.add_argument(
        "--fixation_json_root",
        default=None,
        help="Directory containing fixation-point JSONs. Defaults to outputs/fixations/<dataset>/default.",
    )
    parser.add_argument(
        "--fixation_json_only",
        action="store_true",
        help="Use only fixation points from JSON when a JSON file is present; otherwise fall back to center fixation.",
    )

    args = parser.parse_args()
    if args.fov_index not in [0, 1, 2]:
        print("Invalid fov_index: " + args.fov_index + ". Valid options are [0, 1].")
        exit()
    if args.model_index not in range(0,11):
        print("Invalid model_index: " + args.model_index + ". Valid options are [0, 10].")
        exit()
    type_strs = ['const', 'var']
    if args.type not in type_strs:
        print("Invalid model type: " + args.type + ". Valid options are [const, var].")
        pass
    prefix_idx = type_strs.index(args.type)
    global bypass_filter
    fog_deg_options = [27, 30, 54]
    model_sample_percentage = [3, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    field_of_view_in_degrees = fog_deg_options[args.fov_index]

    num_pool_threads = args.pool_threads

    prefix_idx = type_strs.index(args.type)
    mat_prefix = ['const', 'var'][prefix_idx]

    config = {
        "batchsize": args.batchsize,
        "path": args.path,
        "new_database_path": os.path.join(args.outfolder+'_'+str(field_of_view_in_degrees)+'d_'+str(model_sample_percentage[args.model_index])+'perc', ['Constant', 'Variable'][prefix_idx]),
        "type": args.type,
        "mat_prefix": mat_prefix,
        "prefix_idx": prefix_idx,
        "fixation_json_root": args.fixation_json_root,
        "fixation_json_only": args.fixation_json_only,
        "fp_shift_x": args.fp_shift_x,
        "fp_shift_y": args.fp_shift_y,
    }
    configure_filter_globals(config)

    tmp_org_imgs = sorted(set(os.listdir(old_database_path)))
#    tmp_org_imgs = sorted(set(os.listdir(os.path.join(old_database_path, 'try'))))
    if not os.path.exists(new_database_path):
        os.makedirs(new_database_path)
    images = tmp_org_imgs[(args.index-1)*processing_buffer:np.minimum(len(tmp_org_imgs), args.index*processing_buffer)]

    start = time.time()
    totals = run_images(images, num_pool_threads, config)

    print("Complete. Total elapsed time: " + str(time.time() - start) + " seconds")
    print("Generated: " + str(totals["generated"]) + ", skipped existing: " + str(totals["skipped"]) + ", errors: " + str(totals["errors"]))

if __name__ == "__main__":
    main()
