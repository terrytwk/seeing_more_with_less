import cv2 as cv
import numpy as np
import utils
import math
import torch
from opt_einsum import contract
import os
from pathlib import Path
import scipy.interpolate
import time
import argparse
from PIL import Image
import operator
import json
from datetime import datetime

def relu(x): # Regular ReLU function
    return max(0, x)

def drelu(x, m): # Double ended ReLU function, a linear function between for values between 0 and m, otherwise 0 and m at those values above or below respectively
    return min(relu(x), m)

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
        out_img[:, :, channel][var_indn] = scipy.interpolate.griddata(calc_pts, out_img[:, :, channel][calc_pts[:, 0], calc_pts[:, 1]], new_var_y_x, method='cubic')
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
        help="Output folder name (under .../data/)",
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
        help="Directory containing fixation-point JSONs. Defaults to <path>/filtered/fixation_points.",
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

    bypass_filter = False
    num_pool_threads = args.pool_threads
#    global num_concurrent_img_filters; num_concurrent_img_filters = 6
    global processing_buffer; processing_buffer = args.batchsize
    global old_database_path; old_database_path = args.path
    global new_database_path; new_database_path = os.path.join(args.outfolder+'_'+str(field_of_view_in_degrees)+'d_'+str(model_sample_percentage[args.model_index])+'perc', ['Constant', 'Variable'][prefix_idx])

    prefix_idx = type_strs.index(args.type)
    mat_prefix = ['const', 'var'][prefix_idx]

    global is_full_fov; is_full_fov = False
    tmp_org_imgs = sorted(set(os.listdir(old_database_path)))
#    tmp_org_imgs = sorted(set(os.listdir(os.path.join(old_database_path, 'try'))))
    if not os.path.exists(new_database_path):
        os.makedirs(new_database_path)
    images = tmp_org_imgs[(args.index-1)*processing_buffer:np.minimum(len(tmp_org_imgs), args.index*processing_buffer)]

    start = time.time()
    
    for i in range(len(images)):
        now = datetime.now()
        print(now.strftime("%d/%m/%Y %H:%M:%S") + " Processing image: " + images[i] + " in " + new_database_path)
        fixation_points = []
        fixation_points.append([0, 0])
        obj_ids = []
        obj_ids.append(999)
        if args.type != 'const':
            fixation_json_root = args.fixation_json_root
            if fixation_json_root is None:
                fixation_json_root = os.path.join(old_database_path, 'filtered', 'fixation_points')
            json_info_filename = os.path.join(fixation_json_root, str(Path(images[i]).with_suffix('.json')))
            if os.path.exists(json_info_filename):
                fp_json_file = open(json_info_filename)
                json_data = json.load(fp_json_file)
                image_size = json_data["image_size"]
                objects_info = json_data["objects_info"]
                fp_json_file.close()
            else:
                objects_info = []

            if objects_info and args.fixation_json_only:
                fixation_points = []
                obj_ids = []

            if objects_info:
                for obj_info in objects_info:
                    fixation_points.append([obj_info['centroid'][1],obj_info['centroid'][0]])
                    obj_ids.append(obj_info['obj_id'])

        for ind in range(len(fixation_points)):
            try:
                pil_img = Image.open(os.path.join(old_database_path, images[i]))
                if fixation_points[ind][0] == 0:
                    fixation_points[ind] = [drelu((pil_img.size[0] // 2) + args.fp_shift_x, pil_img.size[0]), drelu((pil_img.size[1] // 2) + args.fp_shift_y, pil_img.size[1])]

                new_image_name = '%s_oid_%d_fpx_%d_fpy_%d.jpg' % (str(Path(images[i]).with_suffix('')), obj_ids[ind], fixation_points[ind][0], fixation_points[ind][1])
                if os.path.exists(os.path.join(new_database_path, new_image_name)):
                    continue

                try:
                    if not ('seconds_per_pixel' in locals()):
                            print("Reading in MAT file")
                            # mat_contents = utils.read_mat('sampling_models_' + str(model_sample_percentage[args.model_index]) + '_samp_per_' + str(field_of_view_in_degrees) + '_fov_deg_unfilt_foveola.mat',                            
                            mat_contents = utils.read_mat('sampling_scheme_params.mat', [
                                'seconds_per_pixel',
                                mat_prefix + '_rf_circles_cntr_xy',
                                mat_prefix + '_rf_filter',
                                mat_prefix + '_rf_roi_half_sz',
                                mat_prefix + '_max_out_img_sz',
                                mat_prefix + '_tar_xlim',
                                mat_prefix + '_tar_ylim',
                                mat_prefix + '_src_xlim',
                                mat_prefix + '_src_ylim',
                            ])
                            print("Finished reading MAT file")

                            seconds_per_pixel = mat_contents['seconds_per_pixel'][0][0]
                            circles_cntr_xy = mat_contents[mat_prefix + '_rf_circles_cntr_xy']

                            global roi_half_sz; roi_half_sz = mat_contents[mat_prefix + '_rf_roi_half_sz'][0][0].astype(int)
                            global calc_pts; calc_pts = roi_half_sz + np.floor(circles_cntr_xy / seconds_per_pixel + 0.5)[:, ::-1].astype(int)
                            global ch1_mode; ch1_mode = True if prefix_idx == 0 else False
                            global filter; filter = mat_contents[mat_prefix + '_rf_filter']
                            global max_out_img_sz; max_out_img_sz = mat_contents[mat_prefix + '_max_out_img_sz'][0][0].astype(int)
                            global tar_xlim; tar_xlim = mat_contents[mat_prefix + '_tar_xlim'].astype(int)
                            global tar_ylim; tar_ylim = mat_contents[mat_prefix + '_tar_ylim'].astype(int)
                            global src_xlim; src_xlim = mat_contents[mat_prefix + '_src_xlim'].astype(int)
                            global src_ylim; src_ylim = mat_contents[mat_prefix + '_src_ylim'].astype(int)
                            mat_contents.clear()
                        
                    out_img, out_img_bbox = filter_preprocessing(images[i], new_image_name, fixation_points[ind])
                except:
                    print("There are some errors generating image: " + new_image_name + ". skipping image...")
            except:
                print("There are some errors loading image: " + images[i] + ". skipping it...")

    print("Complete. Total elapsed time: " + str(time.time() - start) + " seconds")

if __name__ == "__main__":
    main()
