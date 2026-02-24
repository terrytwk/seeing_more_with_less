import sys
import os
import logging
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.objects.main_logger_obj import loggerObj
from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.utils.util_functions import Utilities_helper
from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.objects.recycler_obj import recyclerObj
from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.objects.annotation_processor_obj import annotationProcessor
from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.objects.misk_annotation_processor_obj import miskAnnotationProcessor
from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.objects.prediction_processor_obj import predictionProcessor
from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.objects.tester_obj import testerObj

import argparse

class trialRunnerObj:
    #Some default (usually unnecessary to change) parameters
    _LOG_LEVEL = logging.DEBUG
    _ORIGINAL_ANNOTATIONS_SUBDIR = "original_annotations"
    _PROCESSED_ANNOTATIONS_SAVE_SUBDIR = "filtered_annotations"
    _OVERRIDE_ANNOTATIONS = False
    _MASKRCNN_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[2])
    _FLOW_RUNNER_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))))
    _GENERATED_ANNOTATION_FILES_NAME = "instances_val2017.json"
    _GENERATED_ANNOTATION_FILES_SUMMARIES = "instances_val2017_summary.json"
    _GENERATED_PREDICTIONS_FILES_NAME = "predictions.pth"
    _GENERATED_PREDICTION_FILES_SUMMARIES = "predictions_summary.json"
    _GENERATED_RESULTS_FILE_NAME = "coco_results_original.json"
    _GENERATED_RESULTS_PTH_FILE_NAME = "coco_results_original.pth"
    _GENERATED_RESULTS_BBOX_FILE_NAME = "bbox_original.json"
    _GENERATED_RESULTS_SEGM_FILE_NAME = "segm_original.json"
    _FILES_TO_RECYCLE = [_GENERATED_ANNOTATION_FILES_NAME, _GENERATED_ANNOTATION_FILES_SUMMARIES,
                         _GENERATED_PREDICTIONS_FILES_NAME, _GENERATED_PREDICTION_FILES_SUMMARIES,
                         _GENERATED_RESULTS_BBOX_FILE_NAME, _GENERATED_RESULTS_SEGM_FILE_NAME,
                         _GENERATED_RESULTS_PTH_FILE_NAME, _GENERATED_RESULTS_FILE_NAME]

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

    #Not used currently
    _GENERATED_RESULTS_TXT_FILE_NAME = "coco_results_original.txt"

    def __init__(self, model_name,
                 model_config_file,
                 middle_boundary,
                 bin_threshold,
                 filter_preds,
                 perform_annotation_randomization,
                 sample_ratio_range,
                 sample_ratio_sensitivity,
                 annotation_normalization_large_objects_present,
                 annotation_normalization_factor,
                 annotation_normalization_random_seed,
                 org_annotations_location_w_sample_ratio,
                 images_location,
                 org_predictions_location,
                 experiment_dir,
                 num_trials,
                 current_trial_number,
                 utils_helper):

        self.model_name = model_name
        self.model_config_file = model_config_file
        self.middle_boundary = middle_boundary
        self.bin_threshold = bin_threshold
        self.filter_preds = filter_preds
        self.perform_annotation_randomization = perform_annotation_randomization
        self.sample_ratio_range = sample_ratio_range
        self.sample_ratio_sensitivity = sample_ratio_sensitivity
        self.annotation_normalization_large_objects_present = annotation_normalization_large_objects_present
        self.annotation_normalization_factor = annotation_normalization_factor
        self.annotation_normalization_random_seed = annotation_normalization_random_seed
        self.org_annotations_location_w_sample_ratio = org_annotations_location_w_sample_ratio
        self.images_location = images_location
        self.org_predictions_location = org_predictions_location
        self.experiment_dir = experiment_dir
        self.num_trials = num_trials
        self.current_trial_number = current_trial_number

        self.utils_helper = utils_helper

        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False


    def setup_objects_and_file_structure(self):
        #Setting up logger file structure

        #Setting up the logger
        self.logger = loggerObj(logs_subdir = self.experiment_dir,
                                log_file_name = "log",
                                utils_helper = self.utils_helper,
                                log_level=trialRunnerObj._LOG_LEVEL)
        self.logger.setup_logger()
        logging.info("Finished setting up logger...")
        #---SETUP-BINS---
        self.bins_lower_threshold = [self.bin_threshold]
        self.bins_upper_threshold = [1.0]
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
                                                                      trialRunnerObj._GENERATED_ANNOTATION_FILES_NAME))
            self.generated_predictions_files_paths.append(os.path.join(evaluation_folder,
                                                                       trialRunnerObj._GENERATED_PREDICTIONS_FILES_NAME))


        for lower_threshold, upper_threshold in zip(self.bins_lower_threshold, self.bins_upper_threshold):
            self.generated_test_sets_names.append("coco_2017_" + self.model_name + "_" + str(float(self.bin_threshold)) +
                                                  "_" + str(self.middle_boundary) + "_" +
                                                  str("{:.4f}".format(lower_threshold)) + "_" +
                                                  str("{:.4f}".format(upper_threshold)) + "_eval")

        self.eval_across_bins_csv_file_path = os.path.join(self.experiment_dir,
                                                           trialRunnerObj._GENERATED_HIGH_LVL_CSV_RESULTS_FILE_NAME)
        self.eval_across_bins_graph_file_path = os.path.join(self.experiment_dir,
                                                             trialRunnerObj._GENERATED_HIGH_LVL_GRAPH_FILE_NAME)

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
        logging.info(f"  -  Running normalized annotation eval (misk): {str(self.perform_annotation_randomization)}")
        logging.info(f"  -  Annotation normalization Large objects present (misk): {str(self.annotation_normalization_large_objects_present)}")
        logging.info(f"  -  Annotation normalization random seed (misk): {str(self.annotation_normalization_random_seed)}")
        logging.info(f"  -  Current trial id / total trial id-s (misk): {str(self.current_trial_number)}/{str(self.num_trials-1)}")


    def run_recycler(self, prev_trial_folder):
        #The following function creates an instance of the recycler class,
        #which checks if one of the previous trials has created some files that can be reused in this trial.
        #This is typically the case, for example, with the prediction files for each bin
        if prev_trial_folder is None:
            logging.info("No previous trial folder was passed to the recycler. Not recycling anything, moving on...")
        else:
            logging.info(f"Recycling files from {prev_trial_folder}")
            self.recycler_obj = recyclerObj(prev_trial_folder=prev_trial_folder,
                                            current_folder=self.experiment_dir,
                                            files_to_recycle=trialRunnerObj._FILES_TO_RECYCLE,
                                            current_trial_number=self.current_trial_number)
            self.recycler_obj.copy_subfolders()


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
                annotation_processor_object = annotationProcessor(original_annotations_path= self.org_annotations_location_w_sample_ratio,
                                                                  new_annotations_file_path = gen_annotation_file_path,
                                                                  filter_threshold_array = (lower_threshold, upper_threshold),
                                                                  middle_boundary = self.middle_boundary,
                                                                  sample_ratio_range = self.sample_ratio_range,
                                                                  sample_ratio_sensitivity = self.sample_ratio_sensitivity,
                                                                  utils_helper = self.utils_helper,
                                                                  summary_file_name = trialRunnerObj._GENERATED_ANNOTATION_FILES_SUMMARIES)
                annotation_processor_object.read_annotations()
                annotation_processor_object.filter_annotations_w_wrong_area_and_sample_ratio()
                annotation_processor_object.write_new_annotations_to_disk()
                annotation_processor_object.summarize_annotation_file()
            else: logging.info("Bin annotation file exists. Moving to prediction file processing -->>")
            #---PREDICTION-PROCESSING---
            self.utils_helper.copy_file_from_to(source_path = self.org_predictions_location,
                                                destination_path = gen_prediction_file_path)
            logging.info("Copied original predictions locally.")
            logging.info("NOT performing prediction processing. Moving to evaluation ->>")
            #---------------------------
            #---BIN-EVALUATION---
            if not os.path.exists(os.path.join(evaluation_folder, trialRunnerObj._GENERATED_RESULTS_FILE_NAME)):
                tester_obj = testerObj(model_config_file = self.model_config_file,
                                       current_bin_pth_dir_path = evaluation_folder,
                                       current_bin_annotation_file_path = gen_annotation_file_path,
                                       current_bin_dataset_name = gen_test_set_name,
                                       current_bin_images_path = self.images_location,
                                       utils_helper = self.utils_helper,
                                       results_file_name = trialRunnerObj._GENERATED_RESULTS_FILE_NAME,
                                       results_file_verbose_name = trialRunnerObj._GENERATED_RESULTS_TXT_FILE_NAME)
                tester_obj.build_model()
                #TODO: Add a script which temporarily de-rails print statements to a file
                #so that we see the AR metric
                tester_obj.test_model()
                tester_obj.write_results_to_disk()
                tester_obj.change_result_filename("coco_results.pth", trialRunnerObj._GENERATED_RESULTS_PTH_FILE_NAME)
                tester_obj.change_result_filename("bbox.json", trialRunnerObj._GENERATED_RESULTS_BBOX_FILE_NAME)
                tester_obj.change_result_filename("segm.json", trialRunnerObj._GENERATED_RESULTS_SEGM_FILE_NAME)
            else: logging.info("Evaluation file exists. Moving to next bin (if any) -->>")
            #--------------------
            #-----------------
            self.logger.remove_temp_file_handler_and_add_main_file_handler()
            logging.info(f"Finished working on bin {lower_threshold}-{upper_threshold} in:\n{evaluation_folder}")

        self.summarize_results_csv(self.eval_across_bins_csv_file_path, trialRunnerObj._GENERATED_RESULTS_FILE_NAME,
                                   trialRunnerObj._GENERATED_ANNOTATION_FILES_SUMMARIES)
        self.generate_results_graph_photo(self.eval_across_bins_graph_file_path, self.eval_across_bins_csv_file_path)


    def run_all_misk(self):
        # This function runs:
        # 1. Reading of the smallest number of annotations present in the filt ann files
        # 2. Selects & saves a random subset of all of them in new annotation files
        # 3. Runs the evaluation of the prediction files across the new annotation files
        if self.perform_annotation_randomization:
            logging.info("(Misk) Beginning annotation normalization and new evaluation")
            self.misk_annotation_processor_obj = miskAnnotationProcessor(bins_lower_th_array = self.bins_lower_threshold,
                                                                         bins_upper_th_array = self.bins_upper_threshold,
                                                                         bins_annotations_paths_array = self.generated_annotation_files_paths,
                                                                         bins_paths_array = self.evaluation_folders,
                                                                         ann_summary_file_name = trialRunnerObj._GENERATED_ANNOTATION_FILES_SUMMARIES,
                                                                         utils_helper = self.utils_helper,
                                                                         normalization_factor = self.annotation_normalization_factor,
                                                                         random_seed = self.annotation_normalization_random_seed,
                                                                         large_objects_present = self.annotation_normalization_large_objects_present,
                                                                         ann_subset_files_name_template = trialRunnerObj._GENERATED_ANNOTATION_SUBSAMPLE_FILES_NAME_TMPL,
                                                                         ann_subset_files_summary_name_template = trialRunnerObj._GENERATED_ANNOTATION_SUBSAMPLE_SUMMARY_FILES_NAME_TMPL,
                                                                         current_trial_number=self.current_trial_number,
                                                                         num_trials=self.num_trials)
            self.misk_annotation_processor_obj.read_all_nums_objects()
            self.misk_annotation_processor_obj.eval_normalization_factor()
            self.misk_ann_subsample_size = self.misk_annotation_processor_obj.target_subsample_number
            self.misk_annotation_processor_obj.generate_new_annotation_files_with_subsamples()

            self.normalized_annotations_paths_array = self.misk_annotation_processor_obj.ann_subset_file_paths_array

            #---MISK-VARIABLE-SETTING---
            misk_results_json_filename = trialRunnerObj. \
                _GENERATED_SUBSAMPLE_RESULTS_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_results_txt_filename = trialRunnerObj. \
                _GENERATED_SUBSAMPLE_RESULTS_TXT_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_results_pth_filename = trialRunnerObj. \
                _GENERATED_SUBSAMPLE_RESULTS_PTH_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_results_bbox_filename = trialRunnerObj. \
                _GENERATED_SUBSAMPLE_RESULTS_BBOX_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_results_segm_filename = trialRunnerObj. \
                _GENERATED_SUBSAMPLE_RESULTS_SEGM_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_csv_filename = trialRunnerObj. \
                _GENERATED_SUBSAMPLE_HIGH_LVL_CSV_RESULTS_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)
            misk_graph_photo_filename = trialRunnerObj. \
                _GENERATED_SUBSAMPLE_HIGH_LVL_GRAPH_FILE_NAME_TMPL % str(self.misk_ann_subsample_size)

            self.misk_csv_filepath = os.path.join(self.experiment_dir, misk_csv_filename)
            self.misk_graph_photo_filepath = os.path.join(self.experiment_dir, misk_graph_photo_filename)
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
                    tester_obj.test_model()
                    tester_obj.write_results_to_disk()
                    tester_obj.change_result_filename("coco_results.pth", misk_results_pth_filename)
                    tester_obj.change_result_filename("bbox.json", misk_results_bbox_filename)
                    tester_obj.change_result_filename("segm.json", misk_results_segm_filename)
                else:
                    logging.info("Misk evaluation file exists. Moving to next bin (if any) -->>")

            self.summarize_results_csv(self.misk_csv_filepath, misk_results_json_filename, trialRunnerObj._GENERATED_ANNOTATION_FILES_SUMMARIES)
            self.generate_results_graph_photo(self.misk_graph_photo_filepath, self.misk_csv_filepath)


    def summarize_results_csv(self, eval_across_bins_csv_file_path, potential_results_file_name, potential_ann_summary_file_name):
        if os.path.exists(eval_across_bins_csv_file_path):
            logging.info("CSV file with eval across bins already exists!")
            return

            # Open the CSV file for writing
        with open(eval_across_bins_csv_file_path, "w", newline="") as csv_file:
            # Create a writer object to write to the CSV file
            writer = csv.writer(csv_file)

            # Write the header row to the CSV file
            csv_identification_header = ["lower_bin_thresh", "upper_bin_thresh", "bin_prefix"]
            csv_metrics_header = ["bbox_AP", "bbox_AP50", "bbox_AP75",
                             "bbox_APs", "bbox_APm", "bbox_APl",
                             "bbox_AR@1", "bbox_AR@10", "bbox_AR",
                             "bbox_ARs", "bbox_ARm", "bbox_ARl",
                             "segm_AP", "segm_AP50", "segm_AP75",
                             "segm_APs", "segm_APm", "segm_APl",
                             "segm_AR@1", "segm_AR@10", "segm_AR",
                             "segm_ARs", "segm_ARm", "segm_ARl"]
            csv_ann_distr_headers = ["total_obj", "small_obj", "med_obj",
                             "large_obj"]
            writer.writerow(csv_identification_header + csv_metrics_header + csv_ann_distr_headers)

            for folder in self.evaluation_folders:
                folder_name = os.path.basename(os.path.normpath(folder))
                #Extract the bin from the folder name
                lower_threshold, upper_threshold = self.utils_helper.extract_floats_and_nums_from_string(folder_name)
                potential_eval_storage_file = os.path.join(folder, potential_results_file_name)
                potential_ann_summary_file = os.path.join(folder, potential_ann_summary_file_name)
                assert (os.path.exists(potential_eval_storage_file))
                assert (os.path.exists(potential_ann_summary_file))

                #Extract the value of the evaluation metric
                # Write the data to the CSV file
                try:
                    #Read eval metrics first
                    with open(potential_eval_storage_file) as json_results_file:
                        eval_data = json.load(json_results_file)
                    metric_values = []
                    for metric in csv_metrics_header:
                        metric_dict_recipe = metric.split("_")
                        current_metric = eval_data[metric_dict_recipe[0]][metric_dict_recipe[1]]
                        metric_values.append(current_metric)

                    with open(potential_ann_summary_file) as ann_summary_file:
                        ann_summary_data = json.load(ann_summary_file)
                    ann_summary_lst = [ann_summary_data["after_filtering_annotations_number"],
                                       ann_summary_data["small_annotations"],
                                       ann_summary_data["medium_annotations"],
                                       ann_summary_data["large_annotations"]]

                    to_store_in_csv = [str(lower_threshold), str(upper_threshold),
                                       str(lower_threshold)+"-"+str(upper_threshold)]
                    to_store_in_csv = to_store_in_csv + metric_values + ann_summary_lst
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
        else:
            logging.info("Generating graph photo ...")

        data = pd.read_csv(eval_across_bins_csv_file_path)
        column_names_metrics = list(data.columns)[-28:-4]
        bar_chart_columns = list(data.columns)[-4:]

        # create a 7x4 grid of plots
        fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(16, 28))

        # iterate over the grid of plots and plot each pair of columns
        for i, ax in enumerate(axs.flat):
            # extract the x and y columns for this plot
            x_col = f'lower_bin_thresh'
            x_data = data[x_col].values
            if i < len(column_names_metrics):
                y_col = column_names_metrics[i]
                y_data = data[y_col].values

                slope, intercept = np.polyfit(x_data, y_data, 1)
                line_of_best_fit = slope * x_data + intercept

                # plot the data on the current subplot
                ax.plot(x_data, y_data, marker='o', linestyle='--')
                ax.plot(x_data, line_of_best_fit, '-', linewidth=1.5, color='black', label='L.b.f.')

                # set the title to the name of the y column
                ax.set_title(y_col)

                # hide the x and y axis labels and ticks
                ax.set_xlabel('Bins (lower-thresh)')
                ax.set_ylabel(f'{y_col}')
                ax.set_title('')
            else:
                # plot the data on the current subplot as bar charts
                y_col = bar_chart_columns[i - len(column_names_metrics)]
                y_data = data[y_col].values
                ax.bar(x_data, y_data, width=0.05)

                # set the title to the name of the y column
                ax.set_title(y_col)

                # set the y-axis ticks to show the range of bar heights
                max_height = int(np.ceil(y_data.max()))
                min_height = int(np.floor(y_data.min()))
                num_ticks = 5

                y_ticks = np.asarray(self.utils_helper.generate_equispaced_numbers(min_height,
                                                                                   max_height,
                                                                                   num_ticks))
                ax.set_yticks(y_ticks)

                # hide the x-axis ticks and labels
                ax.set_xlabel('Bins (lower-thresh)')
                ax.set_ylabel(f'{y_col}')
                ax.set_title('')

        # adjust the layout of the subplots
        fig.tight_layout()

        # save the figure to a file
        fig.savefig(eval_across_bins_graph_file_path, dpi=300)
        logging.info(f"Finished generating plot image!")