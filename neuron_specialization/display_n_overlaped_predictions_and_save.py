# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import sys
sys.path.remove('/workspace/object_detection')
sys.path.append('/home/labs/user/userg/Projects/Variable_Resolution/Programming/maskrcnn_multi')


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.coco import COCODataset
from pycocotools.coco import COCO
from demo.predictor_custom import COCO_predictor

import os
import time
import numpy as np
import scipy.misc as sp
from PIL import Image
import matplotlib.pyplot as plt
import tools.evaluation.model_utils as model_tools
import pandas as pd
import itertools


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

def main():

    #Model loading
    custom_config_file_path = "/home/labs/user/userg/Projects/Variable_Resolution/Programming/maskrcnn_multi/configs/5_mod_default_config.yaml"
    custom_weights_dir = "/home/labs/user/userg/Projects/Variable_Resolution/Programming/maskrcnn_multi/trained_models/multi_stacked_resnet101_all_equal/multi_resnet101/last_checkpoint"

    model_tools.setup_env_variables()
    model_predictor = COCO_predictor(cfg=cfg, custom_config_file=custom_config_file_path, \
                                     weight_file_dir=custom_weights_dir, \
                                     confidence_threshold=0.75, use_conf_threshold=True, max_num_pred=23,
                                     min_image_size=60, masks_per_dim=3, choose_top_n_if_none=True,
                                     top_n_if_none=5, top_if_none_critical_number=5)
    print("Model loaded! \n")
    #Model loaded

    annotation_file_path = "/home/labs/user/userh/data/coco_filt/annotations/instances_ch3_val2017.json"
    annotation_file_base_path = os.path.dirname(annotation_file_path)
    images_location_base_dir = "/home/labs/user/userh/data/coco_filt/val2017_multi"
    original_images_location_base_dir = "/home/labs/user/shared/coco/val2017"
    plot_save_dir = "/home/labs/user/userg/Projects/Variable_Resolution/Experiment_visualization/comparative_visualization_min_5/Multi_stacked_resnet101"

    num_channels = 3
    prediction_image_width = 640
    prediction_image_height = 1100
    #DPI is redundant, can be any value
    dpi = 100

    model_name = "Multi_stacked_resnet101"

    overide_folder_content = False

    coco_dataset = COCODataset(annotation_file_path, images_location_base_dir, remove_images_without_annotations=False,
                               num_of_channels=num_channels, transforms=model_predictor.transforms)

    coco = COCO(annotation_file_path)
    img_ids = coco_dataset.ids
    print("Image ids COCO_stack_custom: ", coco_dataset.ids)
    print("Total number of images: ", len(img_ids), "\n")
    images = coco.loadImgs(img_ids)

    errors = []

    for i in range(len(images)):
        #Start timing
        t = time.time()

        img = images[i]

        img_file_path = os.path.join(images_location_base_dir, img['file_name'])
        original_image_path = os.path.join(original_images_location_base_dir, "00"+img['file_name'][1:])
        print("Desired image path: ", img_file_path, "\n")

        new_plot_path = os.path.join(plot_save_dir, img['file_name'])
        if not overide_folder_content:
            if os.path.exists(new_plot_path):
                print("Skipped image already present in folder: ", new_plot_path)
                continue

        try:
            #Load original_image
            img_org = Image.open(img_file_path)
            print("Initial image size: ", img_org.size)
            original_image = Image.open(original_image_path)

            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)

            img_multi_channel, target, idx = coco_dataset.__getitem__(i)
            print("Torch tensor shape: ", list(img_multi_channel.size()))

            # Load tensor and turn into image
            tensor_image = overlay_images_from_multi_ch_tensor(img_multi_channel.permute([1, 2, 0]).numpy())
            print("Overlaid tensor shape: ", np.shape(tensor_image))
            img_overlaid = sp.toimage(tensor_image)
            #Reverse the 3rd dimension's order
            img_overlaid = img_overlaid.resize((img_org.size[0], img_org.size[1]), Image.ANTIALIAS)
            #print("Resized image size: ", img_overlaid.size)
            img_org = img_overlaid

            print("Image ID coco: ", img['id'])
            print("Image ID coco_custom: ", i)

            #Generate model predictions with predeictor_custom
            predictions, predictions_dictionary = model_predictor.run_on_opencv_image(img_multi_channel, img_org)

            # Display prediction and compare to org
            num_plt_rows = 2
            num_plt_cols = 1
            fig, axs = plt.subplots(nrows=num_plt_rows, ncols=num_plt_cols)

            # Set axis off for all subplots
            [axi.set_axis_off() for axi in axs.ravel()]

            fig.suptitle('Model {}'.format(model_name))

            axs[0].imshow(img_org)
            plt.axes(axs[0])
            axs[0].imshow(predictions)

            # Prepare dictionary text box
            plt.axes(axs[1])

            box_annotations = transform_predictions_dict(predictions_dictionary)

            plt.axes(axs[1])
            cell_text = []
            for row in range(len(box_annotations)):
                cell_text.append(box_annotations.iloc[row])

            axs[1].table(cellText=cell_text, colLabels=None, loc='center')

            current_fig_path = os.path.join(plot_save_dir, img['file_name'])

            # Determines the ration of space that each subplot element should take as part of the entire subplot area
            fig.subplots_adjust(top=0.95, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            for axi in axs.ravel():
                # Removes the scales from the axis
                axi.xaxis.set_major_locator(plt.NullLocator())
                axi.yaxis.set_major_locator(plt.NullLocator())

            figure_size_inches_width, figure_size_inches_height = calculate_figure_size_inches(
                prediction_image_width,
                prediction_image_height,
                dpi)
            fig.set_size_inches(figure_size_inches_width, figure_size_inches_height)

            fig.savefig(current_fig_path, bbox_inches='tight',
                        pad_inches=0, dpi=dpi)
            plt.close(fig)

            print('Saved : ', img['file_name'], '\n')
            elapsed = time.time() - t
            print('Elapsed time: ', str(elapsed), '\n')
        except Exception as e:
            e.with_traceback()
            errors.append([e, img['file_name']])
            print("Image ID ", img['file_name'], ": ", e)

    print(errors)


def overlay_images_from_multi_ch_tensor(numpy_tensor_img):
    #print("Numpy shape from FUNCTION: ", np.shape(numpy_tensor_img))

    image_width = np.shape(numpy_tensor_img)[1]
    image_height = np.shape(numpy_tensor_img)[0]
    num_channels = 3
    starting_selection_ch = 0
    separate_channel_averages = np.zeros([image_height, image_width, num_channels], dtype=float)


    for i in range(num_channels):
        print("Color range for image CH{}: ".format(i), get_data_range(list(numpy_tensor_img[:, :, i*3:i*3+3].flatten())))
        current_channel_average = np.average(numpy_tensor_img[:, :, i*3:i*3+3], axis=2)
        separate_channel_averages[:, :, i] = current_channel_average

    #Binarize the separate channel views
    #Third channel is CH3
    separate_channel_averages_bin = (separate_channel_averages != 0).astype(np.int_)
    separate_channel_averages_bin_9_ch = np.zeros([image_height, image_width, 9])
    separate_channel_averages_bin_9_ch[:, :, 6:9] = np.repeat(separate_channel_averages_bin[:, :, 2][:, :, np.newaxis], 3, axis=2)
    separate_channel_averages_bin_9_ch[:, :, 3:6] = np.repeat(separate_channel_averages_bin[:, :, 1][:, :, np.newaxis], 3, axis=2)
    separate_channel_averages_bin_9_ch[:, :, 0:3] = np.repeat(separate_channel_averages_bin[:, :, 0][:, :, np.newaxis], 3, axis=2)

    overlaid_image = np.copy(numpy_tensor_img[:, :, 6:9])
    overlaid_image[separate_channel_averages_bin_9_ch[:, :, 3:6] == 1] = numpy_tensor_img[:, :, 3:6][separate_channel_averages_bin_9_ch[:, :, 3:6] == 1]
    overlaid_image[separate_channel_averages_bin_9_ch[:, :, 0:3] == 1] = numpy_tensor_img[:, :, 0:3][separate_channel_averages_bin_9_ch[:, :, 0:3] == 1]
    overlaid_image = (overlaid_image + 128) / 255
    overlaid_image = np.flip(overlaid_image, axis=2)

    return overlaid_image


def get_data_range(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)


def calculate_figure_size_inches(desired_width_in_pixels, desired_height_in_pixels, dpi):
    width_inches = desired_width_in_pixels / dpi
    height_inches = desired_height_in_pixels / dpi

    return width_inches, height_inches


def transform_predictions_dict(pred_dictionaries):
    print("Transforming prediction dictionary!")

    columns_to_prepare = 3

    #If predictions is < than num_columns, the columns that can't be filled are returned empty!
    prediction_sublists = split_list_into_chunks(pred_dictionaries, columns_to_prepare)

    prediction_dataframe = make_pandas_prediction_dataframe(prediction_sublists)
    prediction_table_string_format = prediction_dataframe.to_string(header = False, index=False)

    return prediction_dataframe


def make_pandas_prediction_dataframe(data_for_columns):
    number_of_columns = len(data_for_columns)

    print("Preparing Pandas dataframe!")
    #print("----DATA FOR COLUMNS----")
    #print(data_for_columns)

    column_names = []
    for i in range(number_of_columns):
        column_names.append("Predictions " + str(i+1))

    #print("Prepared column names: ", column_names)

    prediction_dataframe = pd.DataFrame((_ for _ in itertools.zip_longest(*data_for_columns)), columns=column_names)

    dfStyler = prediction_dataframe.style.set_properties(**{'text-align': 'left'})
    dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    return prediction_dataframe


def split_list_into_chunks(a, n):
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


if __name__ == '__main__':
    main()

