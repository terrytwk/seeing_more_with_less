# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import sys
import os
import subprocess
from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    print(path_main)
    sys.path.append(path_main)
    os.chdir(path_main)
    sys.path.remove('/workspace/object_detection')
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.coco import COCODataset
from pycocotools.coco import COCO
from demo.predictor_custom import COCO_predictor

import os
import time
import copy
import numpy as np
import scipy.misc as sp
from PIL import Image
import matplotlib.pyplot as plt
import EXPERIMENTS.record_and_save_featuremaps_variable.model_utils as model_tools

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
visualisation = {}

#---GLOBAL VARIABLES---
featuremap_pure_ch1 = []
featuremap_pure_ch2 = []
featuremap_pure_ch3 = []
featuremap_mixed_ch1_ch2 = []
featuremap_mixed_ch2_ch3 = []
featuremap_compressor = []
#-----------------

#-----DISTANCE-TO-SELECT-FILTERS-----
filter_number_list = [0, 7, 15, 23, 31, 39, 47, 55]
#--------------------------

def main(args):
    global filter_number_list

    global featuremap_pure_ch1
    global featuremap_pure_ch2
    global featuremap_pure_ch3
    global featuremap_mixed_ch1_ch2
    global featuremap_mixed_ch2_ch3
    global featuremap_compressor

    print("Configuration File:", args.config_file)
    print("tensors Save Directory:", args.tensors_save_dir)
    print("Model Weights Directory:", args.weight_dir)
    print("Evaluation Images Folder:", args.eval_images_folder)
    print("Evaluation Images Annotations:", args.eval_images_annotation)

    #------PREPARE THE VISUALIZATION DIRECTORIES------
    visualization_folders = model_tools.prepare_visualization_folders(tensors_save_dir, filter_number_list)
    #-------------------------------------------------

    #------MODEL & DATASET LOADING--------
    model_tools.setup_env_variables()
    model_predictor = COCO_predictor(cfg=cfg, custom_config_file=config_file, \
                                     weight_file_dir=weight_dir, \
                                     use_conf_threshold=False, max_num_pred=5, min_image_size=60, masks_per_dim=3)
    model = model_predictor.model
    print("Model loaded! \n")
    print(model)
    #Model loaded
    num_channels = 3
    overide_folder_content = False
    coco_dataset = COCODataset(eval_images_annotation, eval_images_folder, remove_images_without_annotations=False,
                               num_of_channels=num_channels, transforms=model_predictor.transforms)
    #-------------------------------------

    #--------HOOK REGISTERING--------
    module_pure_ch1_bn1 = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('pure_ch1_bn1')
    hook_module_pure_ch1_bn1 = module_pure_ch1_bn1.register_forward_hook(hook_pure_ch1)

    module_pure_ch2_bn1 = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('pure_ch2_bn1')
    hook_module_pure_ch2_bn1 = module_pure_ch2_bn1.register_forward_hook(hook_pure_ch2)

    module_pure_ch3_bn1 = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('pure_ch3_bn1')
    hook_module_pure_ch3_bn1 = module_pure_ch3_bn1.register_forward_hook(hook_pure_ch3)

    module_mixed_ch1_ch2_bn1 = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('mixed_ch1_ch2_bn1')
    hook_module_mixed_ch1_ch2_bn1 = module_mixed_ch1_ch2_bn1.register_forward_hook(hook_mixed_ch1_ch2)

    module_mixed_ch2_ch3_bn1 = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('mixed_ch2_ch3_bn1')
    hook_module_mixed_ch2_ch3_bn1 = module_mixed_ch2_ch3_bn1.register_forward_hook(hook_mixed_ch2_ch3)

    #Normalization function of the compressor
    module_compressor_bn = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('bn2')
    hook_module_compressor_bn = module_compressor_bn.register_forward_hook(hook_compressor_bn)
    #--------------------------------

    #---IMAGE FEEDING---
    coco = COCO(eval_images_annotation)
    img_ids = coco_dataset.ids
    print("Image ids COCO_stack_custom: ", coco_dataset.ids)
    print("Total number of images: ", len(img_ids), "\n")
    images = coco.loadImgs(img_ids)

    errors = []

    for i in range(len(images)):
        # Start timing

        img = images[i]

        original_image_path = os.path.join(eval_images_folder, img['file_name'])
        print("Desired image path: ", original_image_path, "\n")

        new_plot_path = os.path.join(visualization_folders[0], img['file_name'])
        if not overide_folder_content:
            if os.path.exists(new_plot_path):
                print("Skipped image already present in folder: ", new_plot_path)
                continue

        try:
            # Load original_image
            original_image = Image.open(original_image_path)
            print("Initial image size: ", original_image.size)

            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)

            img_multi_channel, target, idx = coco_dataset.__getitem__(i)
            print("Torch tensor shape: ", list(img_multi_channel.size()))

            # Load tensor and turn into image
            tensor_img_overlaid = model_tools.overlay_images_from_multi_ch_tensor(img_multi_channel.permute([1, 2, 0]).numpy())
            print("Overlaid tensor shape: ", np.shape(tensor_img_overlaid))
            img_overlaid = sp.toimage(tensor_img_overlaid)
            img_overlaid = img_overlaid.resize((original_image.size[0], original_image.size[1]), Image.ANTIALIAS)
            # print("Resized image size: ", img_overlaid.size)

            print("Image ID coco: ", img['id'])
            print("Image ID coco_custom: ", i)

            # Generate model predictions with predeictor_custom
            predictions, predictions_dictionary = model_predictor.run_on_opencv_image(img_multi_channel, img_overlaid)

            for (folder_current,
                featuremap_pure_ch1_current,
                featuremap_pure_ch2_current,
                featuremap_pure_ch3_current,
                featuremap_mixed_ch1_ch2_current,
                featuremap_mixed_ch2_ch3_current,
                featuremap_compressor_current) in zip(visualization_folders,
                                                     featuremap_pure_ch1,
                                                     featuremap_pure_ch2,
                                                     featuremap_pure_ch3,
                                                     featuremap_mixed_ch1_ch2,
                                                     featuremap_mixed_ch2_ch3,
                                                     featuremap_compressor):

                t = time.time()
                current_filter_num = os.path.basename(os.path.normpath(folder_current))

                # Display prediction and compare to org
                num_plt_rows = 2
                num_plt_cols = 6
                fig, axs = plt.subplots(nrows=num_plt_rows, ncols=num_plt_cols)

                # Set axis off for all subplots
                [axi.set_axis_off() for axi in axs.ravel()]

                fig.suptitle('Image {}'.format(img['file_name']) +"\n Filter " + current_filter_num)

                #---FIRST ROW OF PLOT---
                axs[0, 0].imshow(original_image)
                axs[0, 1].imshow(img_overlaid)
                axs[0, 2].imshow(img_overlaid)
                plt.axes(axs[0, 2])
                coco.showAnns(anns)
                plt.axes(axs[0, 3])
                axs[0, 3].imshow(img_overlaid)
                axs[0, 3].imshow(predictions)
                # Prepare dictionary text box
                plt.axes(axs[0, 5])
                box_annotations = '\n'.join((predictions_dictionary))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axs[0, 5].text(0.5, 0.5, box_annotations, transform=axs[0, 5].transAxes, fontsize=5, \
                               verticalalignment='top', horizontalalignment='center', bbox=props)
                #-----------------------

                #---SECOND ROW OF PLOT---
                #Featuremaps

                axs[1, 0].imshow(featuremap_pure_ch1_current)
                axs[1, 1].imshow(featuremap_pure_ch2_current)
                axs[1, 2].imshow(featuremap_pure_ch3_current)
                axs[1, 3].imshow(featuremap_mixed_ch1_ch2_current)
                axs[1, 4].imshow(featuremap_mixed_ch2_ch3_current)
                axs[1, 5].imshow(featuremap_compressor_current)

                #------------------------

                current_fig_name = os.path.join(folder_current, img['file_name'])
                fig.savefig(current_fig_name, dpi = 700)
                plt.close(fig)

                print('Saved : ', current_fig_name, '\n')
                elapsed = time.time() - t
                print('Elapsed time: ', str(elapsed), '\n')
        except Exception as e:
            e.with_traceback()
            errors.append([e, img['file_name']])
            print("Image ID ", img['file_name'], ": ", e)

    hook_module_pure_ch1_bn1.remove()
    hook_module_pure_ch2_bn1.remove()
    hook_module_pure_ch3_bn1.remove()
    hook_module_mixed_ch1_ch2_bn1.remove()
    hook_module_mixed_ch2_ch3_bn1.remove()

    print(errors)


def display_multi_image_collage(images_info_bundle, plot_size):
    '''
    :param images_info_bundle: Expected to be a Tuple of Tuples, each sub-tuple containing the Title of the image [1]
    and the image itself [0] (in cv2 format)
    :param plot_size: contains number of rows [0] and number of columns [1]
    :return: None
    '''
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = plot_size[0]
    columns = plot_size[1]

    counter = 1

    for image_tuple in images_info_bundle:
        if not image_tuple == ():
            fig.add_subplot(rows, columns, counter)
            # showing image
            plt.imshow(image_tuple[0])
            plt.axis('off')
            plt.title(image_tuple[1])
            counter += 1

    plt.show()


def hook_pure_ch1(m, i, o):
    global filter_number_list
    global featuremap_pure_ch1

    featuremap_pure_ch1_stacked = []

    for filter_number in filter_number_list:
        temp_var = o[0, filter_number, :, :]
        featuremap_pure_ch1_temp = copy.deepcopy(temp_var).cpu().squeeze()
        featuremap_pure_ch1_stacked.append(featuremap_pure_ch1_temp)

    featuremap_pure_ch1 = featuremap_pure_ch1_stacked
    

def hook_pure_ch2(m, i, o):
    global filter_number_list
    global featuremap_pure_ch2

    featuremap_pure_ch2_stacked = []

    for filter_number in filter_number_list:
        temp_var = o[0, filter_number, :, :]
        featuremap_pure_ch2_temp = copy.deepcopy(temp_var).cpu().squeeze()
        featuremap_pure_ch2_stacked.append(featuremap_pure_ch2_temp)

    featuremap_pure_ch2 = featuremap_pure_ch2_stacked
    

def hook_pure_ch3(m, i, o):
    global filter_number_list
    global featuremap_pure_ch3

    featuremap_pure_ch3_stacked = []

    for filter_number in filter_number_list:
        temp_var = o[0, filter_number, :, :]
        featuremap_pure_ch3_temp = copy.deepcopy(temp_var).cpu().squeeze()
        featuremap_pure_ch3_stacked.append(featuremap_pure_ch3_temp)

    featuremap_pure_ch3 = featuremap_pure_ch3_stacked


def hook_mixed_ch1_ch2(m, i, o):
    global filter_number_list
    global featuremap_mixed_ch1_ch2

    featuremap_mixed_ch1_ch2_stacked = []

    for filter_number in filter_number_list:
        temp_var = o[0, filter_number, :, :]
        featuremap_mixed_ch1_ch2_temp = copy.deepcopy(temp_var).cpu().squeeze()
        featuremap_mixed_ch1_ch2_stacked.append(featuremap_mixed_ch1_ch2_temp)

    featuremap_mixed_ch1_ch2 = featuremap_mixed_ch1_ch2_stacked


def hook_mixed_ch2_ch3(m, i, o):
    global filter_number_list
    global featuremap_mixed_ch2_ch3

    featuremap_mixed_ch2_ch3_stacked = []

    for filter_number in filter_number_list:
        temp_var = o[0, filter_number, :, :]
        featuremap_mixed_ch2_ch3_temp = copy.deepcopy(temp_var).cpu().squeeze()
        featuremap_mixed_ch2_ch3_stacked.append(featuremap_mixed_ch2_ch3_temp)

    featuremap_mixed_ch2_ch3 = featuremap_mixed_ch2_ch3_stacked


def hook_compressor_bn(m, i, o):
    global filter_number_list
    global featuremap_compressor

    featuremap_compressor_stack = []

    for filter_number in filter_number_list:
        temp_var = o[0, filter_number, :, :]
        featuremap_compressor_temp = copy.deepcopy(temp_var).cpu().squeeze()
        featuremap_compressor_stack.append(featuremap_compressor_temp)

    featuremap_compressor = featuremap_compressor_stack


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the COCO object detection with configurable paths.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the model's config file.")
    parser.add_argument("--tensors_save_dir", type=str, required=True, help="Directory to save tensors.")
    parser.add_argument("--weight_dir", type=str, required=True, help="Path to the directory where model weights are stored.")
    parser.add_argument("--eval_images_folder", type=str, required=True, help="Folder containing the evaluation images.")
    parser.add_argument("--eval_images_annotation", type=str, required=True, help="Path to the evaluation images' annotations.")

    args = parser.parse_args()
    main(args)