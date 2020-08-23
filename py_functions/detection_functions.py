from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import py_functions.model_functions as mf
import numpy as np
import json
import os


# ----------------------------
# utility functions
# ----------------------------

# draws all detected objects' bounding boxes for one photo
def draw_all_boxes(saturation_map, datalist):
    for object_data in datalist:
        object_dims = object_data["Bounding box"]
        mf.draw_box(saturation_map, object_dims)


# rewrites object detection dictionary so it's separated by directories
def dict_by_results_dir(full_dict):
    dir_dict = defaultdict(lambda: {})
    for photo in full_dict:
        dir_path = full_dict[photo][0][1]
        dir_dict[dir_path][photo] = full_dict[photo]
    return dir_dict


# reads and returns needed data for specified object from dictionary with object detection data
def read_data(data_dict, category_index, data_index, width, height):
    object_score = float(data_dict["detection_scores"][data_index])
    object_class = int(data_dict["detection_classes"][data_index])
    object_name = category_index[object_class]["name"]
    object_dims = mf.calculate_box_dimensions(data_dict["detection_boxes"][data_index], width, height)

    return object_score, object_class, object_name, object_dims


# builds dictionary for one photo, for writing into JSON files
def build_json_dict(data_dict, category_index, photo, width, height, score_threshold):
    # list for objects' data
    datalist = []
    for i in range(len(data_dict["detection_classes"])):
        # read values from data_dict
        object_score, object_class, object_name, object_dims = read_data(data_dict, category_index, i,
                                                                         width, height)
        # skip current object if its score is under threshold
        if object_score < score_threshold:
            continue
        # create dictionary and append it into datalist
        object_data = {
            "Class id": object_class,
            "Class name": object_name,
            "Score": object_score,
            "Bounding box": object_dims
        }
        datalist.append(object_data)

    # dictionary for current photo, to be written into JSON file
    return {"Photo": photo, "Width": width, "Height": height, "Objects": datalist}


# ----------------------------
# main function
# ----------------------------

# recursively runs detection on photos in given directory
# and saves new results into given results directory
def run_detection(model, model_name, category_index, photos_path, results_path, save_photos,
                  score_threshold=0.5, apply_nms=False, use_yolo=False, suppression_threshold=0.3):
    full_dict = {}
    for item in sorted(os.listdir(photos_path)):
        # run recursively in subdirectory
        if os.path.isdir(os.path.join(photos_path, item)):
            new_photos_path = os.path.join(photos_path, item)
            new_results_path = os.path.join(results_path, item)
            full_dict = dict(full_dict, **run_detection(model, model_name, category_index, new_photos_path,
                                                        new_results_path, save_photos, score_threshold, apply_nms,
                                                        use_yolo, suppression_threshold))
        # check if file jpg/png
        elif item[-4:] == ".jpg" or item[-4:] == ".png":
            # get name of photo without file extension, for saving results files with identical name
            # also add model name
            filename = "{} - {}".format(item[:-4], model_name)

            # open photo as numpy array
            item_path = os.path.join(photos_path, item)
            with Image.open(item_path) as opened_image:
                photo_np = np.array(opened_image)

            # reshape grayscale image into 3 channels (tensorflow model expects RGB image)
            if len(photo_np.shape) < 3:
                photo_np = np.stack((photo_np,) * 3, axis=-1)

            # dimensions
            photo_x, photo_y = photo_np.shape[1], photo_np.shape[0]

            # result photos subdir
            photos_results_path = results_path + "/result_photos"
            if not os.path.exists(photos_results_path):
                os.makedirs(photos_results_path)

            # add original file extension to result photo filename
            result_filename = filename + item[-4:]
            # where result photo will be saved
            result_photo_path = os.path.join(photos_results_path, result_filename)

            # execute object detection
            print("Current photo:", item_path)
            if not use_yolo:
                data_dict = mf.run_image_inference(model, photo_np,
                                                   apply_nms=apply_nms, nms_threshold=suppression_threshold)
            else:
                data_dict = mf.detect_with_yolo(model, photo_np, suppression_threshold)

            if save_photos:
                # save photo with bounding boxes
                results_photo = mf.get_inference_image(photo_np, data_dict, category_index,
                                                       line_thickness=2, score_threshold=score_threshold)
                img = Image.fromarray(results_photo)
                print("    Saving visualized result on location", result_photo_path)
                img.save(result_photo_path)

            photo_data = (filename, results_path, photo_x, photo_y)
            full_dict[item_path] = (photo_data, data_dict)

    return full_dict


# ----------------------------
# function for generating additional data files
# ----------------------------

# generates JSON files for each photo
# can also generate saturation maps + saturation statistics if specified
def generate_data_files(full_dict, category_index, score_threshold=0.5, generate_saturation_stats=True):
    for photo in full_dict:
        # read data
        filename, results_path, photo_x, photo_y = full_dict[photo][0]
        data_dict = full_dict[photo][1]

        # subdir for detection data
        data_results_path = results_path + "/detection_data"
        if not os.path.exists(data_results_path):
            os.makedirs(data_results_path)

        print("Current photo:", photo)
        # photo info for writing into JSON file
        photo_info = build_json_dict(data_dict, category_index, photo, photo_x, photo_y, score_threshold)

        # dump data into JSON file
        print("    Generating JSON file...")
        data_file = os.path.join(data_results_path, filename + ".json")  # same name as original photo
        print("    Writing data into", data_file)
        with open(data_file, "w+") as f:
            json.dump(photo_info, f, indent=4)

        # save saturation stats if necessary
        if generate_saturation_stats:
            # variables for saving saturation maps
            saturation_maps_path = results_path + "/saturation_maps"
            if not os.path.exists(saturation_maps_path):
                os.makedirs(saturation_maps_path)
            # image array for saturation map
            saturation_map = np.zeros((photo_y, photo_x))
            # save saturation map
            print("    Generating saturation map...")
            # draw boxes on saturation map
            objects_list = photo_info["Objects"]
            draw_all_boxes(saturation_map, objects_list)

            # list for image occupation statistics
            stats_list = mf.calculate_saturation_stats(saturation_map)

            # normalize array to have values from 0 to 255
            saturation_map = mf.normalize_grayscale(saturation_map)
            # saturation map must be saved as png to avoid JPEG color compression
            map_file = os.path.join(saturation_maps_path, filename + ".png")  # same name as original photo
            print("    Saving saturation map into", map_file)
            sat_img = Image.fromarray(saturation_map, mode="L")
            sat_img.save(map_file)

            # save saturation stats file
            print("    Generating saturation stats text file...")
            # build string for saturation statistics
            stats_str = mf.format_saturation_stats(stats_list, photo, photo_x, photo_y)
            # write saturation stats into .txt file
            text_file = os.path.join(saturation_maps_path, filename + ".txt")  # same name as original photo
            print("    Writing saturation statistics into", text_file)
            with open(text_file, "w+") as f:
                f.write(stats_str)


# ----------------------------
# functions for generating CSV files
# ----------------------------

# generates combined CSV file with all detection data for all photos in detection data dictionary
def generate_csv(full_dict, category_index, model_name, results_path,
                 score_thresh=0.5, csv_filename="all_photos_data"):
    # add model name to filename
    csv_filename = "{} - {}.csv".format(csv_filename, model_name)
    # path where CSV file will be saved
    csv_path = os.path.join(results_path, csv_filename)
    # list with all rows for CSV file
    # one row == one photo
    csv_rows = []
    # counter for max detected objects at specified score threshold
    # used for appropriate header length
    max_objects = 0
    # first, generate rows for each photo
    for photo in full_dict:
        photo_x, photo_y = full_dict[photo][0][2:]
        data_dict = full_dict[photo][1]
        row_str = ""
        num_objects = 0
        # for each detected object
        for i in range(len(data_dict["detection_classes"])):
            object_score, object_class, object_name, object_dims = read_data(data_dict, category_index, i,
                                                                             photo_x, photo_y)

            # skip if confidence score too low
            if object_score < score_thresh:
                continue
            num_objects += 1

            # read values for bounding box
            box_width = object_dims["width"]
            box_height = object_dims["height"]
            min_x = object_dims["min x"]
            max_x = object_dims["max x"]
            min_y = object_dims["min y"]
            max_y = object_dims["max y"]
            # build string with values for current object
            object_str = ";{};{};{};{};{};{};{};{};{}".format(object_class, object_name, object_score,
                                                              box_width, box_height,
                                                              min_x, max_x, min_y, max_y)
            # add object string to full row string
            row_str += object_str

        # each row contains photo's properties before data for objects
        row_prefix = "{};{};{};{}".format(photo, photo_x, photo_y, num_objects)
        row_str = row_prefix + row_str
        if num_objects > max_objects:
            max_objects = num_objects

        # append built row string into list with all rows
        csv_rows.append(row_str)

    # first part of header row
    header = "photo path;width;length;detected objects"
    # part of header with columns for one object
    object_part_header = ";object class;object name;confidence score;object width;object height;" \
                         "min x;max x;min y;max y"
    # make header as long as needed for max number of objects
    header += object_part_header * max_objects

    # write everything into CSV file
    with open(csv_path, "w+") as f:
        f.write(header)
        for row in csv_rows:
            f.write("\n" + row)


# generates corresponding combined CSV file for each directory with photos
def generate_csv_by_folder(full_dict, category_index, model_name, score_thresh=0.5):
    dir_dict = dict_by_results_dir(full_dict)
    for dir_path in dir_dict:
        generate_csv(dir_dict[dir_path], category_index, model_name, dir_path, score_thresh,
                     csv_filename="dir_photos_data")


# ----------------------------
# functions for generating JSON files
# ----------------------------

# generates combined JSON file with all detection data for all photos in detection data dictionary
def generate_json(full_dict, category_index, model_name, results_path, score_thresh=0.5,
                  json_filename="all_photos_data"):
    # add model name to filename
    json_filename = "{} - {}.json".format(json_filename, model_name)

    # path for JSON file
    json_path = os.path.join(results_path, json_filename)
    # list with JSON dictionaries for each photo
    photos_data = []

    # get JSON dictionaries for each photo
    for photo in full_dict:
        photo_x, photo_y = full_dict[photo][0][2:]
        data_dict = full_dict[photo][1]
        # get list with objects' data and append it into full list
        photo_info = build_json_dict(data_dict, category_index, photo, photo_x, photo_y, score_thresh)
        photos_data.append(photo_info)
    # write list with all dictionaries into JSON file
    with open(json_path, "w+") as f:
        json.dump(photos_data, f, indent=4)


# generates corresponding combined JSON file for each directory with photos
def generate_json_by_folder(full_dict, category_index, model_name, score_thresh=0.5):
    dir_dict = dict_by_results_dir(full_dict)
    for dir_path in dir_dict:
        generate_json(dir_dict[dir_path], category_index, model_name, dir_path, score_thresh,
                      json_filename="dir_photos_data")


# ----------------------------
# functions for generating bar charts for confidence score thresholds
# ----------------------------

# generates lists for bar charts for confidence score thresholds
# there are 10 thresholds - 0%-90%, incremented by 10%
def score_charts(full_dict):

    # 2D list (matrix) for number of detected objects by different confidence score thresholds
    # 1st index: score threshold --> item at nth index == list for bar chart at threshold >= 10n
    # 2nd index: number of detected objects -> item in row at nth index == number of photos with n detected objects
    all_charts = np.zeros(shape=(10, 11)).astype(int)

    # for each photo on which detection was executed
    for photo in full_dict:
        objects_dict = full_dict[photo][1]  # data dict for current photo
        num_objects = len(objects_dict["detection_scores"])  # number of detected objects

        # list of numbers of detected objects at specific threshold, at the start filled with num detected objects
        objects_by_thresholds = np.array([num_objects] * 10)

        # for each detected objects
        for score in objects_dict["detection_scores"]:
            # index for first threshold, that is bigger than threshold for current object
            # example: score == 0.15 --> threshold index == int(score * 10) == 1, next threshold index == 2
            next_threshold_index = int(score * 10) + 1

            # do it only if current object doesn't already fall into the last (biggest) threshold
            if next_threshold_index < 9:
                # subtract current object from all thresholds that are bigger than actual threshold for its score
                difference = np.ones(len(objects_by_thresholds) - next_threshold_index).astype(int)
                objects_by_thresholds[next_threshold_index:] -= difference

        # increment appropriate values in lists for bar charts
        for i in range(len(objects_by_thresholds)):
            detected_objects = objects_by_thresholds[i]
            if detected_objects > 10:
                # that means number of detected objects == 10+
                detected_objects = 10
            all_charts[i][detected_objects] += 1

    return all_charts


# saves all 10 score threshold charts
def save_score_charts(full_dict, model_name, results_path):
    chart_list = score_charts(full_dict)
    base_path = os.path.join(results_path, "confidence_threshold_charts")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for i in range(len(chart_list)):
        y_values = chart_list[i]
        x_values = [str(x) for x in range(len(y_values))]
        x_values[-1] = "10+"
        score_threshold = i * 10
        filename = "at least {} % - {}.png".format(score_threshold, model_name)
        file_path = os.path.join(base_path, filename)
        title = "Detected objects with confidence score >= {} %".format(score_threshold)
        plt.bar(x_values, y_values)
        plt.xticks(x_values)
        plt.title(title)
        plt.xlabel("Number of detected objects")
        plt.ylabel("Number of photographs")
        plt.savefig(file_path)
        plt.close()


# saves individual score charts for each subdirectory
def save_score_charts_by_folder(full_dict, model_name):
    dir_dict = dict_by_results_dir(full_dict)
    for dir_path in dir_dict:
        save_score_charts(dir_dict[dir_path], model_name, dir_path)
