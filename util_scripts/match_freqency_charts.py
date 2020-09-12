import py_functions.comparison_functions as cf
import matplotlib.pyplot as plt
import numpy as np
import os


"""
This script is used for plotting bar charts with comparison match frequencies.
It can be used for plotting bar chart for one IOU threshold
or it can be used for plotting grouped bar chart for multiple IOU thresholds at once.
"""

# directory with results
res_dir = "../Results/2020-08-21 Booking.com_photos"

# which models will be used for comparison
model1 = "rfcn_resnet101_coco"
model2 = "ssd_mobilenet_v1_fpn_coco"

# filename for chart image
save_filename = "iou_grouped_chart {} with {}.png".format(model1, model2)

# directory where file will be saved
save_dir = os.path.join(res_dir, "iou_charts")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# full path to saved image
save_file_path = os.path.join(save_dir, save_filename)

# base path for json
base_json_path = os.path.join(res_dir, "{}", "all_photos_data - {}.json")
# json paths for both models
m1_json = base_json_path.format(model1, model1)
m2_json = base_json_path.format(model2, model2)

# comparison dicts for different IOU thresholds
comparison_dict_05 = cf.compare_two_models((model1, model2), (m1_json, m2_json), 0.5)
comparison_dict_07 = cf.compare_two_models((model1, model2), (m1_json, m2_json), 0.7)
comparison_dict_085 = cf.compare_two_models((model1, model2), (m1_json, m2_json), 0.85)
iou_thresholds = [0.5, 0.7, 0.85]

# get pairs for each comparison dict
pairs = cf.get_common_location_pairs(comparison_dict_05)
pairs2 = cf.get_common_location_pairs(comparison_dict_07)
pairs3 = cf.get_common_location_pairs(comparison_dict_085)


# function that does actual chart generation
def bar_charts_from_freqs(pairs_list, file_path, m_names, iou, use_multiple=False):
    # for multiple iou thresholds (grouped bar chart),
    # create wider graph for better readability
    if use_multiple:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig, ax = plt.subplots()

    # use logarithmic scale on y axis so smaller values will also be visible
    plt.yscale("log")
    # set axis limits
    plt.xlim(right=140)
    plt.ylim(top=1000)
    pair_tuples1 = sorted([(pair, pairs_list[0].count(pair))
                           for pair in set(pairs_list[0])],
                          key=lambda x: -x[1])
    y1 = [pair[1] for pair in pair_tuples1]

    x_values = ["-".join(pair[0]) for pair in pair_tuples1]
    x_loc = np.arange(len(x_values))  # the label locations
    ax.set_yscale("log")
    if use_multiple:
        width = 0.3
        pair_tuples2 = sorted([(pair, pairs_list[1].count(pair))
                               for pair in set(pairs_list[1])],
                              key=lambda x: -x[1])
        pair_tuples3 = sorted([(pair, pairs_list[2].count(pair))
                               for pair in set(pairs_list[2])],
                              key=lambda x: -x[1])
        y2 = [pair[1] for pair in pair_tuples2]
        yy = len(y1) - len(y2)
        y2.extend([0] * yy)
        y3 = [pair[1] for pair in pair_tuples3]
        yyy = len(y1) - len(y3)
        y3.extend([0] * yyy)
        ax.bar(x_loc - width, y1, width)
        ax.bar(x_loc, y2, width)
        ax.bar(x_loc + width, y3, width)
        ax.set_title("IOU thresholds comparison\nwhen comparing {}\nwith {}".format(m_names[0], m_names[1]))
    else:
        ax.bar(x_loc, y1)
        ax.set_title("{}\nwith {}\nat IoU={}".format(m_names[0], m_names[1], iou))
    ax.set_xlabel("Matches")
    ax.set_ylabel("Match frequencies")
    plt.xticks([])
    if use_multiple:
        plt.legend(["IOU threshold = {}".format(iou[0]),
                    "IOU threshold = {}".format(iou[1]),
                    "IOU threshold = {}".format(iou[2])])
    plt.savefig(file_path, bbox_inches="tight", dpi=400)
    plt.close()


# generate chart file
bar_charts_from_freqs([pairs, pairs2, pairs3], save_file_path,
                      (model1, model2), iou_thresholds, use_multiple=True)
