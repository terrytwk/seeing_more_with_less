import sys
import os
import logging
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    print(path_main)
    sys.path.append(path_main)
    os.chdir(path_main)
    sys.path.remove('/workspace/object_detection')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.logger_obj import loggerObj
from EXPERIMENTS.complete_bin_evaluation_pipeline.utils.util_functions import Utilities_helper
from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.annotation_processor_obj import annotationProcessor
from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.misk_annotation_processor_obj import miskAnnotationProcessor
from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.prediction_processor_obj import predictionProcessor
from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.tester_obj import testerObj

import argparse

class flowRunner:
    #Some default (usually unnecessary to change) parameters
    _LOG_LEVEL = logging.DEBUG
    _ORIGINAL_ANNOTATIONS_SUBDIR = "original_annotations"
    _PROCESSED_ANNOTATIONS_SAVE_SUBDIR = "filtered_annotations"
    _OVERRIDE_ANNOTATIONS = False
    _MASKRCNN_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    _FLOW_RUNNER_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))))
    _GENERATED_ANNOTATION_FILES_NAME = "instances_val2017.json"
    _GENERATED_ANNOTATION_FILES_SUMMARIES = "instances_val2017_summary.json"
    _GENERATED_PREDICTIONS_FILES_NAME = "predictions.pth"
    _GENERATED_PREDICTION_FILES_SUMMARIES = "predictions_summary.json"
    _GENERATED_RESULTS_FILE_NAME = "coco_results_original.json"
    _GENERATED_RESULTS_TXT_FILE_NAME = "coco_results_original.txt"
    _GENERATED_RESULTS_PTH_FILE_NAME = "coco_results_original.pth"
    _GENERATED_RESULTS_BBOX_FILE_NAME = "bbox_original.json"
    _GENERATED_RESULTS_SEGM_FILE_NAME = "segm_original.json"
    _GENERATED_HIGH_LVL_CSV_RESULTS_FILE_NAME = "eval_across_bins.csv"
    _GENERATED_HIGH_LVL_GRAPH_FILE_NAME = "performance_graph.png"
    #This parameter determines whether the script will filter predictions having masks with no Logit score > 0.5
    #If FALSE: predictions regardless of their mask logits score will be kept
    #If TRUE: only predictions having at least 1 mask logit score > 0.5 will be kept
    _FILTER_MASK_LOGITS = False

    #Misk functionality defaults
    #Those variables are made to have an indicator at their end, to signify against how many
    #objects was the model evaluated (in annotation file)
    _GENERATED_ANNOTATION_SUBSAMPLE_FILES_NAME_TMPL = "instances_val2017_%s.json"
    _GENERATED_ANNOTATION_SUBSAMPLE_SUMMARY_FILES_NAME_TMPL = "instances_val2017_%s_summary.json"
    _GENERATED_SUBSAMPLE_RESULTS_FILE_NAME_TMPL = "coco_results_on_%s.json"
    _GENERATED_SUBSAMPLE_RESULTS_TXT_FILE_NAME_TMPL = "coco_results_on_%s.txt"
    _GENERATED_SUBSAMPLE_RESULTS_PTH_FILE_NAME_TMPL = "coco_results_on_%s.pth"
    _GENERATED_SUBSAMPLE_RESULTS_BBOX_FILE_NAME_TMPL = "bbox_on_%s.json"
    _GENERATED_SUBSAMPLE_RESULTS_SEGM_FILE_NAME_TMPL = "segm_on_%s.json"
    _GENERATED_SUBSAMPLE_HIGH_LVL_CSV_RESULTS_FILE_NAME_TMPL = "eval_across_bins_on_%s.csv"
    _GENERATED_SUBSAMPLE_HIGH_LVL_GRAPH_FILE_NAME_TMPL = "performance_graph_on_%s.png"

    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for complete resolution-bin evaluation pipeline')
        parser.add_argument('-mn', '--model-name', nargs='?',
                            type=str,
                            default = "variable_resolution_pretrained_resnet_norm",
                            required = False,
                            help='This name will be used as: '
                                 '1. Name of the sub-directory in which the experiment files will be stored'
                                 '2. Prefix to the log file')
        parser.add_argument('-mcf', '--model-config-file', nargs='?',
                            type=str,
                            default = os.path.join(flowRunner._MASKRCNN_PARENT_DIR_ABSOLUTE,
                                                   "configs/R-101-FPN/variable_pretrained_resnet/variable_pretrained_resnet_baseline_resnet_norm.yaml"),
                            required = False,
                            help='This parameter is used: '
                                 '1. For constructing the model during testing'
                                 '2. NOT for its test set: this parameter is irrelevant')
        parser.add_argument('-mb', '--middle-boundary', nargs='+',
                            required=False,
                            type = int,
                            default=[100],
                            help='The edge size of the middle square we define to have high-resolution')
        parser.add_argument('-bs', '--bin-spacing', nargs='?',
                            type=float,
                            default=0.04,
                            required = False,
                            help='(% / 100) The space between each resolution bin.'
                                 'E.g. If this paramter is set to 0.1, one can expct that the paradigm'
                                 'will generate 10 and evaluate 10 bins, starting from 0.0-0.1 and ending with'
                                 '0.9-1.0'
                                 'IMPORTANT: This parameter is also appended to the name of the'
                                 'folder in which this experiment is stored')
        parser.add_argument('-fp', '--filter-preds', nargs='?',
                            type=str,
                            default="False",
                            required = False,
                            help='Whether to filter the prediction files (True),'
                                 ' or only the annotation files (False).'
                                 'This measure was implemented due to suspected bias in the eval'
                                 'stemming from the different number of objects in each pred. bin')
        parser.add_argument('-pan', '--perform-annotation-normalization', nargs='?',
                            type=str,
                            default="False",
                            required = False,
                            help='The annotation normalization includes taking the smallest number of objects'
                                 'present in each annotation file after filtering, and selecting from ALL'
                                 'other annotation files a random subset of the same number of objects.'
                                 'Thereafter, preforming the evaluations on those subsets.')
        parser.add_argument('-anf', '--annotation-normalization-factor', nargs='?',
                            default=0.9,
                            type=float,
                            required = False,
                            help='The smallest number of annotations present in single bin will'
                                 'be the size of the subsample we randomly extract from each other bin'
                                 'in order to normalize the annotations. However, we multiply this size by the'
                                 'normalization ratio: so that there is some randomness in the smallest'
                                 ' annotation also. '
                                 'However, this parameter can also be an integer > 1. Then: exactly that '
                                 'many random objects will be picked from each annotation bin')
        parser.add_argument('-oal', '--org-annotations-location', nargs='?',
                            type=str,
                            default = os.path.normpath(os.path.join(flowRunner._MASKRCNN_PARENT_DIR_ABSOLUTE,
                                                   "annotations/original_annotations/instances_val2017.json")),
                            required = False,
                            help='The location of the original annotation file to be filtered')
        parser.add_argument('-il', '--images-location', nargs='?',
                            type=str,
                            default = "-",
                            required = False,
                            help='The location of the images for the parent dataset'
                                 'E.g. The Variable images')
        parser.add_argument('-opl', '--org-predictions-location', nargs='?',
                            type=str,
                            default = os.path.normpath(os.path.join(flowRunner._MASKRCNN_PARENT_DIR_ABSOLUTE,
                                                   "trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/coco_2017_variable_val/predictions.pth")),
                            required = False,
                            help='The location of the original prediction file to be filtered')
        parser.add_argument('-psl', '--parent-storage-location', nargs='?',
                            type=str,
                            default = os.path.normpath(os.path.join(flowRunner._FLOW_RUNNER_PARENT_DIR_ABSOLUTE,
                                                   "evaluations")),
                            required = False,
                            help='The location in which the newly generated annotation file'
                                 ' as well as the newly generated predictions file will be stored')
        parser.add_argument('-efi', '--experiment-folder-identificator', nargs='?',
                            type=str,
                            default = "ann_norm",
                            required = False,
                            help='As the amount of tunable parameters of this script grows, it is needed'
                                 'to have some idea of what an experiment folder contains. This variable'
                                 'allows one to append any string the the final experiment folder name')

        self.args = parser.parse_args()
        assert(self.args.perform_annotation_normalization == "True" or
               self.args.perform_annotation_normalization == "False")
        assert(self.args.filter_preds == "True" or
               self.args.filter_preds == "False")

        self.model_name = self.args.model_name
        self.model_config_file = self.args.model_config_file
        self.middle_boundary = self.args.middle_boundary[0]
        self.bin_spacing = self.args.bin_spacing
        self.filter_preds = True if self.args.filter_preds == "True" else False
        self.perform_annotation_norm = True if self.args.perform_annotation_normalization == "True" else False
        self.annotation_normalization_factor = self.args.annotation_normalization_factor
        self.org_annotations_location = self.args.org_annotations_location
        self.images_location = self.args.images_location
        self.org_predictions_location = self.args.org_predictions_location
        self.parent_storage_location = self.args.parent_storage_location
        self.experiment_folder_identificator = self.args.experiment_folder_identificator

        self.experiment_name = self.model_name + "_" + str(float(self.bin_spacing)) + "_" +\
                               str(self.middle_boundary) + "_" + self.experiment_folder_identificator

        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False


    def setup_objects_and_file_structure(self):
        self.utils_helper = Utilities_helper()
        #Setting up logger file structure
        self.experiment_dir = os.path.join(self.parent_storage_location, self.experiment_name)
        self.utils_helper.check_dir_and_make_if_na(self.experiment_dir)

        #Setting up the logger
        self.logger = loggerObj(logs_subdir = self.experiment_dir,
                                log_file_name = "log",
                                utils_helper = self.utils_helper,
                                log_level=flowRunner._LOG_LEVEL)
        logging.info("Finished setting up logger...")
        logging.info("Passed arguments -->>")
        logging.info('\n  -  '+ '\n  -  '.join(f'{k}={v}' for k, v in vars(self.args).items()))
        #---SETUP-BINS---
        self.bins_lower_threshold = list(np.around(np.linspace(0, 1-self.bin_spacing,
                                                               int(1/self.bin_spacing)),
                                                   decimals=4))
        self.bins_upper_threshold = list(np.around(np.linspace(self.bin_spacing, 1,
                                                               int(1 / self.bin_spacing)),
                                                   decimals=4))
        assert len(self.bins_upper_threshold) == len(self.bins_lower_threshold)
        logging.info("Bin pairs setup complete -->>")

        #---SETUP-EVALUATION-FOLDER-NAMES---
        self.evaluation_folders = []
        for lower_threshold, upper_threshold in zip(self.bins_lower_threshold, self.bins_upper_threshold):
            _current_dir = os.path.join(self.experiment_dir, str("{:.4f}".format(lower_threshold))
                                        + "_" + str("{:.4f}".format(upper_threshold))
                                        + "_eval")
            _current_dir = os.path.normpath(_current_dir)
            self.utils_helper.check_dir_and_make_if_na(_current_dir)
            self.evaluation_folders.append(_current_dir)
        logging.info("Setup individual bin evaluation folders -->>")
        #-----------------------------------

        #---SETUP-GENERATED-FILES-PATHS---
        #This portion of the script generates the complete paths each new annotation and predictions file will assume
        self.generated_annotation_files_paths = []
        self.generated_predictions_files_paths = []
        self.generated_test_sets_names = []
        for evaluation_folder in self.evaluation_folders:
            self.generated_annotation_files_paths.append(os.path.join(evaluation_folder,
                                                                      flowRunner._GENERATED_ANNOTATION_FILES_NAME))
            self.generated_predictions_files_paths.append(os.path.join(evaluation_folder,
                                                                       flowRunner._GENERATED_PREDICTIONS_FILES_NAME))


        for lower_threshold, upper_threshold in zip(self.bins_lower_threshold, self.bins_upper_threshold):
            self.generated_test_sets_names.append("coco_2017_" + self.model_name + "_" + str(float(self.bin_spacing)) +
                                                  "_" + str(self.middle_boundary) + "_" +
                                                  str("{:.4f}".format(lower_threshold)) + "_" +
                                                  str("{:.4f}".format(upper_threshold)) + "_eval")

        self.eval_across_bins_csv_file_path = os.path.join(self.experiment_dir,
                                                           flowRunner._GENERATED_HIGH_LVL_CSV_RESULTS_FILE_NAME)
        self.eval_across_bins_graph_file_path = os.path.join(self.experiment_dir,
                                                             flowRunner._GENERATED_HIGH_LVL_GRAPH_FILE_NAME)

        #---------------------------------
        logging.info('\n  -  '+ '\n  -  '.join(f'({l} | {u}) \n  -  Evaluation dir: {f}'
                                               f' \n  -  Annotation file: {a}'
                                               f' \n  -  Predictions file: {p}'
                                               f' \n  -  Test set name: {t}' for l, u, f, a, p, t in
                                               zip(self.bins_lower_threshold,
                                                   self.bins_upper_threshold,
                                                   self.evaluation_folders,
                                                   self.generated_annotation_files_paths,
                                                   self.generated_predictions_files_paths,
                                                   self.generated_test_sets_names)))
        logging.info(f"  -  CSV file with eval across bins: {self.eval_across_bins_csv_file_path}")
        logging.info(f"  -  Filtering predictions: {str(self.filter_preds)}")
        logging.info(f"  -  Running normalized annotation eval (misk): {str(self.perform_annotation_norm)}")


    def run_all_vanilla(self):
        # A function which runs the typical per-bin evaluation:
        # 1. Filter annotation files
        # 2. Filter pred files (or not, based on args.don-filter-preds)
        # 3. Run classic evaluation with filtered components
        logging.info("Running per bin evaluation -->>")

        for lower_threshold, upper_threshold,\
            evaluation_folder, gen_annotation_file_path,\
            gen_prediction_file_path, gen_test_set_name in zip(self.bins_lower_threshold,
                                                                     self.bins_upper_threshold,
                                                                     self.evaluation_folders,
                                                                     self.generated_annotation_files_paths,
                                                                     self.generated_predictions_files_paths,
                                                                     self.generated_test_sets_names):
            logging.info(f"Working on bin {lower_threshold}-{upper_threshold} in:\n{evaluation_folder}")
            self.logger.add_temp_file_handler_and_remove_main_file_handler(evaluation_folder)
            logging.info("Test: should be in bin-specific dir only")
            #---ANNOTATION-PREP---
            if not os.path.exists(gen_annotation_file_path):
                annotation_processor_object = annotationProcessor(original_annotations_path= self.org_annotations_location,
                                                                  new_annotations_file_path = gen_annotation_file_path,
                                                                  filter_threshold_array = (lower_threshold, upper_threshold),
                                                                  middle_boundary = self.middle_boundary,
                                                                  utils_helper = self.utils_helper,
                                                                  summary_file_name = flowRunner._GENERATED_ANNOTATION_FILES_SUMMARIES)
                annotation_processor_object.read_annotations()
                annotation_processor_object.filter_annotations_w_wrong_area_ratio()
                annotation_processor_object.write_new_annotations_to_disk()
                annotation_processor_object.summarize_annotation_file()
            else: logging.info("Bin annotation file exists. Moving to prediction file processing -->>")
            #---PREDICTION-PROCESSING---
            if not os.path.exists(gen_prediction_file_path):
                prediction_processor_object = predictionProcessor(
                    org_predictions_location = self.org_predictions_location,
                    new_predictions_path = gen_prediction_file_path,
                    images_location = self.images_location,
                    annotation_file_location = gen_annotation_file_path,
                    area_threshold_array = (lower_threshold, upper_threshold),
                    middle_boundary = self.middle_boundary,
                    filter_preds = self.filter_preds,
                    model_cfg_path = self.model_config_file,
                    utils_helper = self.utils_helper,
                    mask_logit_threshold = 0.5 if flowRunner._FILTER_MASK_LOGITS else 0.0,
                    summary_file_name = flowRunner._GENERATED_PREDICTION_FILES_SUMMARIES)
                prediction_processor_object.setup_objects_and_misk_variables()
                prediction_processor_object.read_predictions()
                prediction_processor_object.filter_predictions_w_wrong_area_ratio()
                prediction_processor_object.write_new_predictions_to_disk()
                prediction_processor_object.summarize_prediction_file()
            else: logging.info("Bin prediction file exists. Moving to evaluation -->>")
            logging.info("Finished prediction file processing ->>")
            #---------------------------
            #---BIN-EVALUATION---
            if not os.path.exists(os.path.join(evaluation_folder, flowRunner._GENERATED_RESULTS_FILE_NAME)):
                tester_obj = testerObj(model_config_file = self.model_config_file,
                                       current_bin_pth_dir_path = evaluation_folder,
                                       current_bin_annotation_file_path = gen_annotation_file_path,
                                       current_bin_dataset_name = gen_test_set_name,
                                       current_bin_images_path = self.images_location,
                                       utils_helper = self.utils_helper,
                                       results_file_name = flow_runner._GENERATED_RESULTS_FILE_NAME,
                                       results_file_verbose_name = flow_runner._GENERATED_RESULTS_TXT_FILE_NAME)
                tester_obj.build_model()
                #TODO: Add a script which temporarily de-rails print statements to a file
                #so that we see the AR metric
                tester_obj.test_model()
                tester_obj.write_results_to_disk()
                tester_obj.change_result_filename("coco_results.pth", flowRunner._GENERATED_RESULTS_PTH_FILE_NAME)
                tester_obj.change_result_filename("bbox.json", flowRunner._GENERATED_RESULTS_BBOX_FILE_NAME)
                tester_obj.change_result_filename("segm.json", flowRunner._GENERATED_RESULTS_SEGM_FILE_NAME)
            else: logging.info("Evaluation file exists. Moving to next bin (if any) -->>")
            #--------------------
            #-----------------
            self.logger.remove_temp_file_handler_and_add_main_file_handler()
            logging.info(f"Finished working on bin {lower_threshold}-{upper_threshold} in:\n{evaluation_folder}")

        self.summarize_results_csv(self.eval_across_bins_csv_file_path, flowRunner._GENERATED_RESULTS_FILE_NAME)
        self.generate_results_graph_photo(self.eval_across_bins_graph_file_path, self.eval_across_bins_csv_file_path)


    def run_all_misk(self):
        # This function runs:
        # 1. Reading of the smallest number of annotations present in the filt ann files
        # 2. Selects & saves a random subset of all of them in new annotation files
        # 3. Runs the evaluation of the prediction files across the new annotation files
        if self.perform_annotation_norm:
            self.misk_annotation_processor_obj = miskAnnotationProcessor(bins_lower_th_array = self.bins_lower_threshold,
                                                                         bins_upper_th_array = self.bins_upper_threshold,
                                                                         bins_annotations_paths_array = self.generated_annotation_files_paths,
                                                                         bins_paths_array = self.evaluation_folders,
                                                                         ann_summary_file_name = flowRunner._GENERATED_ANNOTATION_FILES_SUMMARIES,
                                                                         utils_helper = self.utils_helper,
                                                                         normalization_factor = self.annotation_normalization_factor,
                                                                         ann_subset_files_name_template = flowRunner._GENERATED_ANNOTATION_SUBSAMPLE_FILES_NAME_TMPL,
                                                                         ann_subset_files_summary_name_template = flowRunner._GENERATED_ANNOTATION_SUBSAMPLE_SUMMARY_FILES_NAME_TMPL)
            self.misk_annotation_processor_obj.read_all_nums_objects()
            self.misk_annotation_processor_obj.eval_normalization_factor()
            self.misk_ann_subsample_size = self.misk_annotation_processor_obj.target_subsample_number
            self.misk_annotation_processor_obj.generate_new_annotation_files_with_subsamples()

            self.normalized_annotations_paths_array = self.misk_annotation_processor_obj.ann_subset_file_paths_array

            #---MISK-VARIABLE-SETTING---
            misk_results_json_filename = flowRunner. \
                _GENERATED_SUBSAMPLE_RESULTS_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_results_txt_filename = flowRunner. \
                _GENERATED_SUBSAMPLE_RESULTS_TXT_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_results_pth_filename = flowRunner. \
                _GENERATED_SUBSAMPLE_RESULTS_PTH_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_results_bbox_filename = flowRunner. \
                _GENERATED_SUBSAMPLE_RESULTS_BBOX_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_results_segm_filename = flowRunner. \
                _GENERATED_SUBSAMPLE_RESULTS_SEGM_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_csv_filename = flowRunner. \
                _GENERATED_SUBSAMPLE_HIGH_LVL_CSV_RESULTS_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_graph_photo_filename = flowRunner. \
                _GENERATED_SUBSAMPLE_HIGH_LVL_GRAPH_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)

            misk_csv_filepath = os.path.join(self.experiment_dir, misk_csv_filename)
            misk_graph_photo_filepath = os.path.join(self.experiment_dir, misk_graph_photo_filename)
            #------------------------

            #---RUN-TESTING-SCRIPT---
            for lower_threshold, upper_threshold, \
                evaluation_folder, gen_annotation_file_path, \
                gen_prediction_file_path, gen_test_set_name in zip(self.bins_lower_threshold,
                                                                   self.bins_upper_threshold,
                                                                   self.evaluation_folders,
                                                                   self.normalized_annotations_paths_array,
                                                                   self.generated_predictions_files_paths,
                                                                   self.generated_test_sets_names):
                logging.info(f"Working on bin {lower_threshold}-{upper_threshold} in:\n{evaluation_folder}")
                self.logger.add_temp_file_handler_and_remove_main_file_handler(evaluation_folder)
                logging.info("Proceeding to normalized annotation evaluation ...")
                # ---------------------------
                # ---BIN-EVALUATION---
                if not os.path.exists(os.path.join(evaluation_folder, misk_results_json_filename)):
                    tester_obj = testerObj(model_config_file=self.model_config_file,
                                           current_bin_pth_dir_path=evaluation_folder,
                                           current_bin_annotation_file_path=gen_annotation_file_path,
                                           current_bin_dataset_name=gen_test_set_name,
                                           current_bin_images_path=self.images_location,
                                           utils_helper=self.utils_helper,
                                           results_file_name=misk_results_json_filename,
                                           results_file_verbose_name=misk_results_txt_filename)
                    tester_obj.build_model()
                    # TODO: Add a script which temporarily de-rails print statements to a file
                    # so that we see the AR metric
                    tester_obj.test_model()
                    tester_obj.write_results_to_disk()
                    tester_obj.change_result_filename("coco_results.pth", misk_results_pth_filename)
                    tester_obj.change_result_filename("bbox.json", misk_results_bbox_filename)
                    tester_obj.change_result_filename("segm.json", misk_results_segm_filename)
                else:
                    logging.info("Misk evaluation file exists. Moving to next bin (if any) -->>")

            self.summarize_results_csv(misk_csv_filepath, misk_results_json_filename)
            self.generate_results_graph_photo(misk_graph_photo_filepath, misk_csv_filepath)


    def summarize_results_csv(self, eval_across_bins_csv_file_path, potential_results_file_name):
        if os.path.exists(eval_across_bins_csv_file_path):
            logging.info("CSV file with eval across bins already exists!")
            return

            # Open the CSV file for writing
        with open(eval_across_bins_csv_file_path, "w", newline="") as csv_file:
            # Create a writer object to write to the CSV file
            writer = csv.writer(csv_file)

            # Write the header row to the CSV file
            writer.writerow(["lower_bin_thresh", "upper_bin_thresh", "bin_prefix", "AP"])
            for folder in self.evaluation_folders:
                folder_name = os.path.basename(os.path.normpath(folder))
                #Extract the bin from the folder name
                lower_threshold, upper_threshold = self.utils_helper.extract_floats_and_nums_from_string(folder_name)
                potential_eval_storage_file = os.path.join(folder, potential_results_file_name)
                assert(os.path.exists(potential_eval_storage_file))

                #Extract the value of the evaluation metric
                # Write the data to the CSV file
                try:
                    with open(potential_eval_storage_file) as json_results_file:
                        json_data = json.load(json_results_file)
                    avg_precision = json_data["bbox"]["AP"]
                    to_store_in_csv = [str(lower_threshold), str(upper_threshold),
                                       str(lower_threshold)+"-"+str(upper_threshold),
                                       avg_precision]
                    writer.writerow(to_store_in_csv)
                except Exception as e:
                    logging.critical(f"Error received while generating the .CSV file: {e.with_traceback()}")
                    return
        logging.info("Finished generating high-level .csv file")


    def generate_results_graph_photo(self, eval_across_bins_graph_file_path, eval_across_bins_csv_file_path):
        # This function takes the generated .csv file and outputs a photo of the model performance graph
        if os.path.exists(eval_across_bins_graph_file_path):
            logging.info("CSV file with eval across bins already exists!")
            return

        data = pd.read_csv(eval_across_bins_csv_file_path)
        # Get the x and y data from the Pandas DataFrame
        x_data = data.iloc[:, 0]  # first column
        y_data = data.iloc[:, 3]  # fourth column

        # Create a new figure and axis for the plot
        fig, ax = plt.subplots()
        # Plot the data as a line plot
        ax.plot(x_data, y_data, marker='o', linestyle='--')
        # Set the axis labels and title
        ax.set_xlabel('Bins (lower-thresh)')
        ax.set_ylabel('AP (IoU=0.50:0.95), maxDets=100')
        ax.set_title('Performance graph')
        # Save the plot as a PNG image
        fig.savefig(eval_across_bins_graph_file_path, dpi=300, bbox_inches='tight')
        logging.info(f"Finished generating plot image!")


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.setup_objects_and_file_structure()
    flow_runner.run_all_vanilla()
    flow_runner.run_all_misk()