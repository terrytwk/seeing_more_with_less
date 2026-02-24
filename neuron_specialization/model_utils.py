# Function containing custom NN model related functions.

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data.transforms import build_transforms
from PIL import Image
from apex import amp

import os
import sys
import torch
import cv2
from torch.autograd import Variable
#import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

def prepare_visualization_folders(base_dir, filter_number_list):
    new_folders_paths = []

    for current_folder_num in filter_number_list:
        current_folder_num_str = str(current_folder_num)
        new_folder_path = os.path.join(base_dir, current_folder_num_str)

        if os.path.exists(new_folder_path):
            new_folders_paths.append(new_folder_path)
            continue
        else:
            os.mkdir(new_folder_path)
            print("Directory '% s' created" % new_folder_path)
            new_folders_paths.append(new_folder_path)

    return new_folders_paths

def overlay_images_from_multi_ch_tensor(numpy_tensor_img):
    # print("Numpy shape from FUNCTION: ", np.shape(numpy_tensor_img))

    image_width = np.shape(numpy_tensor_img)[1]
    image_height = np.shape(numpy_tensor_img)[0]
    num_channels = 3
    starting_selection_ch = 0
    separate_channel_averages = np.zeros([image_height, image_width, num_channels], dtype=float)

    for i in range(num_channels):
        print("Color range for image CH{}: ".format(i),
              get_data_range(list(numpy_tensor_img[:, :, i * 3:i * 3 + 3].flatten())))
        current_channel_average = np.average(numpy_tensor_img[:, :, i * 3:i * 3 + 3], axis=2)
        separate_channel_averages[:, :, i] = current_channel_average

    # Binarize the separate channel views
    # Third channel is CH3
    separate_channel_averages_bin = (separate_channel_averages != 0).astype(np.int_)
    separate_channel_averages_bin_9_ch = np.zeros([image_height, image_width, 9])
    separate_channel_averages_bin_9_ch[:, :, 6:9] = np.repeat(separate_channel_averages_bin[:, :, 2][:, :, np.newaxis],
                                                              3, axis=2)
    separate_channel_averages_bin_9_ch[:, :, 3:6] = np.repeat(separate_channel_averages_bin[:, :, 1][:, :, np.newaxis],
                                                              3, axis=2)
    separate_channel_averages_bin_9_ch[:, :, 0:3] = np.repeat(separate_channel_averages_bin[:, :, 0][:, :, np.newaxis],
                                                              3, axis=2)

    overlaid_image = np.copy(numpy_tensor_img[:, :, 6:9])
    overlaid_image[separate_channel_averages_bin_9_ch[:, :, 3:6] == 1] = numpy_tensor_img[:, :, 3:6][
        separate_channel_averages_bin_9_ch[:, :, 3:6] == 1]
    overlaid_image[separate_channel_averages_bin_9_ch[:, :, 0:3] == 1] = numpy_tensor_img[:, :, 0:3][
        separate_channel_averages_bin_9_ch[:, :, 0:3] == 1]
    overlaid_image = (overlaid_image + 128) / 255
    overlaid_image = np.flip(overlaid_image, axis=2)

    return overlaid_image

def get_data_range(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)

def load_model(cutom_config_file, weight_file_dir):
    #Sets the used GPU to id 0
    default_weight_file_name = "model_final.pth"

    cfg.merge_from_file(cutom_config_file)
    cfg.freeze()

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    #Weight loading
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=weight_file_dir)
    _ = checkpointer.load(os.path.join(weight_file_dir, default_weight_file_name))

    model.eval()

    return model

def setup_env_variables():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"