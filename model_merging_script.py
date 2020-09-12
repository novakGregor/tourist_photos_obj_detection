from collections import defaultdict
from itertools import combinations
import py_functions.merging_functions as mf
import py_functions.comparison_functions as cf
import os

"""
This script is used for merging arbitrary combinations of models and their evaluation.
It can also be used for evaluating individual models, if we are "merging" only one model
"""

# determines how combinations will be made
# "all_models" -> use all models
# "coco_dataset" -> create combinations of 3 and 2
# "openimages_dataset" -> create combinations of 2
used_combinations = "all_models"

# which models will be used in combinations
# !!! IMPORTANT !!!
# for using coco dataset only, you must return openimages models manually!
# similarly for openimages!
used_models = (
    ("f_rcnn_c", "faster_rcnn_resnet50_coco"),
    ("rfcn_c", "rfcn_resnet101_coco"),
    ("ssd_c", "ssd_mobilenet_v1_fpn_coco"),
    ("YOLOv3_c", "YOLOv3"),
    ("f_rcnn_oid", "faster_rcnn_inception_resnet_v2_atrous_oidv4"),
    ("ssd_oid", "ssd_mobilenetv2_oidv4")
)

# json path with ground truth annotations for test photos
ground_truth_json = "data/openimages_test_small_ground_truth.json"
# directory with results data for merging models
base_results_dir = "Results/openimages_test_small"

path_to_merged_models = os.path.join(base_results_dir, "merged_models", used_combinations)
if not os.path.exists(path_to_merged_models):
    os.makedirs(path_to_merged_models)

tab_dir = path_to_merged_models + "/tab_files/non-pruned"
tab_dir_pruned = path_to_merged_models + "/tab_files/pruned"
for dir_path in [tab_dir, tab_dir_pruned]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

count_threshold = 10
jaccard_threshold = 0.015

# for coco dataset, we are merging models in combinations of 3 and 2
if used_combinations == "coco_dataset":
    combinations_r3 = list(combinations(used_models, 3))
    combinations_r2 = list(combinations(used_models, 2))
    all_combinations = combinations_r3 + combinations_r2 + [used_models]
# for openimages dataset, we only have two models, therefore only combinations of 2
elif used_combinations == "openimages_dataset":
    all_combinations = list(combinations(used_models, 2))
# for individual models, we have combinations of 1
elif used_combinations == "individual_models":
    all_combinations = list(combinations(used_models, 1))
# else, use all models
else:
    all_combinations = [used_models]


# function for saving tabs with evaluating data
# i.e., comparison with ground truth + precision and recall for each found matching pair
def save_merged_tabs(model_names, model_jsons, save_paths, count_thresh, jaccard_thresh):

    # compare with ground truth and retrieve location pairs
    comparison_dict = cf.compare_two_models(model_names, model_jsons)
    location_pairs = cf.get_common_location_pairs(comparison_dict)

    # get object counts
    obj_count1 = comparison_dict["object_count1"]
    obj_count2 = comparison_dict["object_count2"]
    obj_counts = (obj_count1, obj_count2)

    # unpack save paths
    save_path, save_path_thresh = save_paths

    # get tuples for saving into tab file
    loc_tuples = cf.location_pair_tuples(location_pairs, obj_counts, False, 0, 0.0)
    loc_tuples_thresh = cf.location_pair_tuples(location_pairs, obj_counts, False, count_thresh, jaccard_thresh)

    # retrieve detected counts for test dataset, for each object category
    # i.e., amount of objects from test dataset that were actually detected
    detected_names2 = defaultdict(int)
    for row in loc_tuples:
        # row[-1] -> ground truth object name
        # row[1] -> ground truth count
        detected_names2[row[-1]] += row[1]

    # header for tab file
    header = ["index", "count_detected_name", "detected_name",
              "num_matches", "Jaccard_index", "matches_by_true_count", "matches_by_detected_count",
              "true_name", "count_true_name", "count_detected_true_name"]

    # save tab files with and without thresholds
    for tuples, path in ((loc_tuples, save_path), (loc_tuples_thresh, save_path_thresh)):
        with open(path, "w+") as f:
            f.write("\t".join(header) + "\n")
            for row in range(len(tuples)):
                name1, matches, jacc_index, name2 = tuples[row]
                cnt1 = obj_count1[name1]
                cnt2 = obj_count2[name2]
                true_detected_count = detected_names2[name2]
                matches_by_true_count = matches / cnt2
                matches_by_detected_count = matches / cnt1
                if matches_by_detected_count > 1:
                    matches_by_detected_count = 1
                row_val_list = [row, cnt1, name1,
                                matches, jacc_index, matches_by_true_count, matches_by_detected_count,
                                name2, cnt2, true_detected_count]
                f.write("\t".join(map(str, row_val_list)) + "\n")


# execute merging for every combination
num_combinations = len(all_combinations)
for i in range(num_combinations):
    print("Current combination: {}/{}".format(i + 1, num_combinations))
    comb = all_combinations[i]
    short_names = [m[0] for m in comb]
    full_names = [m[1] for m in comb]
    merged_name = "-".join(short_names)
    merged_model_json = os.path.join(path_to_merged_models, merged_name+".json")
    print(merged_model_json)

    # save merged data
    joined_models = mf.join_models(full_names, base_results_dir)
    combined_objects = mf.combine_objects_data(joined_models, 0.5)
    mf.merge_combined_objects_data(combined_objects, merged_model_json)

    tab_path = tab_dir + "/{}.tab".format(merged_name)
    tab_path_pruned = tab_dir_pruned + "/{}.tab".format(merged_name)

    tab_paths = (tab_path, tab_path_pruned)
    m_names = (merged_name, "openimages_test_small_ground_truth", )
    m_jsons = (merged_model_json, ground_truth_json)

    # save evaluation tabs
    save_merged_tabs(m_names, m_jsons, tab_paths,
                     count_threshold, jaccard_threshold)
