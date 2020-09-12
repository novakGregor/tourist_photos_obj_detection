import py_functions.model_functions as mf
import py_functions.detection_functions as df
import matplotlib.pyplot as plt
import numpy as np
import json
import os


"""
this script is used for generating bar charts for overall image occupation,
as well as bar charts for image "saturation" (charts for object overlapping).
"""


# Retrieves full saturation statistics from specified model's detection data json
def get_saturation_stats(detection_json):
    full_stats_list = []
    with open(detection_json) as f:
        detection_dict = json.load(f)

    for photo in detection_dict:
        width = photo["Width"]
        height = photo["Height"]
        all_pixels = width * height
        objects = photo["Objects"]
        saturation_map = np.zeros((height, width))
        df.draw_all_boxes(saturation_map, objects)
        stats_list = mf.calculate_saturation_stats(saturation_map)
        stats_list = [(value / all_pixels) * 100 for value in stats_list]
        full_stats_list.append(stats_list)

    return full_stats_list


# creates list for image occupation chart and image overlap charts
def get_stats_bar_charts(saturation_stats):
    all_charts = np.zeros(shape=(11, 11)).astype(int)
    for photo_list in saturation_stats:
        all_percentage = sum(photo_list[1:])
        all_idx = int(all_percentage / 10) + 1
        if all_percentage == 0:
            all_idx = 0
        if all_idx == 11:
            all_idx = 10
        all_charts[0][all_idx] += 1
        for i in range(1, 11):
            percentage_value = photo_list[i]
            percentage_idx = int(percentage_value / 10)
            if percentage_idx == 11:
                percentage_idx = 10
            all_charts[i][percentage_idx] += 1
    return all_charts


# saves all image occupation chart and all image overlap charts
def save_coverage_charts(model_json, model_name, results_path):
    saturation_stats = get_saturation_stats(model_json)
    chart_list = get_stats_bar_charts(saturation_stats)
    for i in range(len(chart_list)):
        y_values = chart_list[i]
        x_values = ["0%", "1-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        if i > 0:
            filename = "image_occupation_{}_object(s) - {}.png".format(i, model_name)
            title = "Image occupation for {} object(s) at once".format(i)
        else:
            filename = "overall_image_occupation - {}.png".format(model_name)
            title = "Overall image occupation"

        file_path = os.path.join(results_path, filename)
        plt.bar(x_values, y_values)
        plt.xticks(x_values, rotation=45)
        plt.title(title)
        plt.xlabel("Percentage of image occupied")
        plt.ylabel("Number of photographs")
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()


# generate charts for all models in specified results dir
results_dir = "../Results/Booking.com_photos/models"
for used_model in os.listdir(results_dir):
    if used_model == "comparison":
        continue
    print("Current model:", used_model)
    base_json_file = results_dir + "/{}/all_photos_data - {}.json"
    json_file = base_json_file.format(used_model, used_model)
    charts_save_dir = results_dir + "/{}/image_occupation_charts".format(used_model)
    if not os.path.exists(charts_save_dir):
        os.makedirs(charts_save_dir)

    save_coverage_charts(json_file, used_model, charts_save_dir)
