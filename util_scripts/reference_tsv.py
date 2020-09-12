from collections import defaultdict
import py_functions.comparison_functions as cf
import csv
import os


"""
this script is used for generating reference TSV (TAB) file for detection data
of specified merged models system
"""


# path to where detection data for used model system is located
results_base_path = "../Results/Booking.com_photos/merged_models/best_system"
system_name = "rfcn_c-YOLOv3_c-f_rcnn_oid"

# evaluation tab file for used model system
tab_file = "{}.tab".format(system_name)

# specify paths to pruned and non-prund tab files
# and create tuple with those two paths
tab_path = os.path.join(results_base_path, tab_file)
tab_path_pruned = os.path.join(results_base_path, "pruned_" + tab_file)
tab_paths = (tab_path, tab_path_pruned)

# json file with detection data for used model system
json_file = "all_photos_data - {}.json".format(system_name)
json_path = os.path.join(results_base_path, json_file)

# where reference tsv file will be saved
save_path = os.path.join(results_base_path, "info_file.tab")


# reads data from evaluation tsv (tab) file
# and returns it as dictionary
def read_dict_from_tsv(eval_tab):
    tsv_dict = defaultdict(list)
    with open(eval_tab) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            detected_name = row[2]
            detected_recall = row[5]
            detected_precision = row[6]
            true_name = row[7]
            tsv_dict[detected_name].append((true_name, detected_precision, detected_recall))

    for name in tsv_dict:
        tsv_dict[name] = sorted(tsv_dict[name], key=lambda x: -float(x[1]))

    return tsv_dict


# builds and saves reference TSV (TAB) file
def build_info_tsv(model_json, eval_tabs, save_file):
    full_tsv_dict = read_dict_from_tsv(eval_tabs[0])
    pruned_tsv_dict = read_dict_from_tsv(eval_tabs[1])

    json_tuple = (model_json, model_json)
    comparison_dict = cf.compare_two_models(json_tuple, json_tuple, iou_thresh=1.0)
    obj_count = comparison_dict["object_count1"]

    obj_rows = []
    for obj_name in obj_count:
        obj_freq = obj_count[obj_name]
        # if object is located in pruned tsv file, use all found matching pairs from pruned file
        if obj_name in pruned_tsv_dict:
            data_tuple = pruned_tsv_dict[obj_name]
        # else use all matching pairs from pruned file
        else:
            data_tuple = full_tsv_dict[obj_name]
        # if object name does not appear in detection data on test dataset, specify as "no_info"
        if not data_tuple:
            data_tuple = ["no_info"]
        # filter matching pairs so only the most relevant are used
        if len(data_tuple) > 5:
            data_tuple = [data for data in data_tuple if float(data[1]) >= 0.01 and float(data[2]) >= 0.01]

        obj_row = [obj_name, obj_freq, data_tuple]
        obj_rows.append(obj_row)

    obj_rows = sorted(obj_rows, key=lambda x: -x[1])
    file_header = ["detected_name", "detected_count", "possible_detections (name, precision, recall)"]
    with open(save_file, "w+") as f:
        f.write("\t".join(file_header) + "\n")
        for obj in obj_rows:
            f.write("\t".join(map(str, obj)) + "\n")


data_dict = read_dict_from_tsv(tab_path)
build_info_tsv(json_path, tab_paths, save_path)
