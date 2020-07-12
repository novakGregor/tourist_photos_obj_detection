from object_detection.utils import label_map_util
from datetime import datetime
from PIL import Image
import model_functions as mf
import numpy as np
import json
import time
import os

# dict with name strings for models from TensorFlow object detection zoo
models = {
    "ssd": "ssd_mobilenet_v1_coco_2018_01_28",
    "f_rcnn": "faster_rcnn_nas_coco_2018_01_28"
}

# ----------------------------
# variables for running object detection
# ----------------------------

used_model = models["ssd"]
# actual model for object detection
model = mf.load_tensorflow_model(used_model)

# for differing results folder according to date
today = datetime.today().strftime("%Y-%m-%d")

# directory from where photos will be used
photos_dir = "Photos/Piran_en"
# directory where results will be saved
results_dir = os.path.join("Results", today + " Piran_en", used_model)

# list of the strings that is used to add correct label for each box
PATH_TO_LABELS = "Data/mscoco_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ----------------------------
# main function
# ----------------------------

# recursively runs detection on photos in given directory
# and saves new results into given results directory
def run_detection(photos_path, results_path, score_threshold=0.5, use_yolo=False, suppression_threshold=0.3):
    full_dict = {}
    for item in os.listdir(photos_path):
        # run recursively in subdirectory
        if os.path.isdir(os.path.join(photos_path, item)):
            new_photos_path = os.path.join(photos_path, item)
            new_results_path = os.path.join(results_path, item)
            full_dict = dict(full_dict, **run_detection(new_photos_path, new_results_path,
                                                        score_threshold, use_yolo, suppression_threshold))
        # check if file jpg/png
        elif item[-4:] == ".jpg" or item[-4:] == ".png":
            # get name of photo without file extension, for saving results files with identical name
            filename = item[:-4]

            # open photo as numpy array
            item_path = os.path.join(photos_path, item)
            photo_np = np.array(Image.open(item_path))

            # dimensions
            photo_x, photo_y = photo_np.shape[1], photo_np.shape[0]

            # result photos subdir
            photos_results_path = results_path + "/result_photos"
            if not os.path.exists(photos_results_path):
                os.makedirs(photos_results_path)

            # where result photo will be saved
            result_photo_path = os.path.join(photos_results_path, item)

            # execute object detection
            print("Current photo:", item_path)
            if not use_yolo:
                data_dict = mf.run_image_inference(model, photo_np)
            else:
                data_dict = mf.detect_with_yolo(model, photo_np, score_threshold, suppression_threshold)

            # save photo with bounding boxes
            results_photo = mf.get_inference_image(item_path, data_dict, category_index,
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

def generate_data_files(full_dict, score_threshold=0.5, generate_saturation_stats=True):
    for photo in full_dict:
        # read data
        filename, results_path, photo_x, photo_y = full_dict[photo][0]
        data_dict = full_dict[photo][1]

        # subdir for detection data
        data_results_path = results_path + "/detection_data"
        if not os.path.exists(data_results_path):
            os.makedirs(data_results_path)

        # list for objects' data
        datalist = []

        # variables for saving saturation maps
        saturation_maps_path = None
        saturation_map = None
        stats_list = None
        # only initialize them if necessary
        if generate_saturation_stats:
            saturation_maps_path = results_path + "/saturation_maps"
            if not os.path.exists(saturation_maps_path):
                os.makedirs(saturation_maps_path)
            # image array for saturation map
            saturation_map = np.zeros(shape=(photo_y, photo_x, 3)).astype(np.uint8)
            saturation_map.fill(255)  # blank white photo
            # list for counting pixel colors occurrences
            stats_list = [0] * 11  # 0-10+ objects

        # for each recognized object
        print("Current photo:", photo)
        for i in range(len(data_dict["detection_classes"])):
            # get score and convert it into normal float (can't write numpy types into JSON)
            object_score = float(data_dict["detection_scores"][i])
            # skip current object if its score is under threshold
            if object_score < score_threshold:
                continue
            object_class = int(data_dict["detection_classes"][i])
            # actual name of object, not its ID
            object_name = category_index[object_class]["name"]
            # get bounding box dimensions
            object_dims = mf.calculate_box_dimensions(data_dict["detection_boxes"][i], photo_x, photo_y)
            # create dictionary and append it into datalist
            object_data = {
                "Class id": object_class,
                "Class name": object_name,
                "Score": object_score,
                "Bounding box": object_dims
            }
            datalist.append(object_data)

            if generate_saturation_stats:
                # draw box on saturation map
                mf.draw_box(saturation_map, object_dims, stats_list)

        # photo info for writing into JSON file
        photo_info = {"Photo": photo, "Width": photo_x, "Height": photo_y}

        # combine photo info and objects' data into one dictionary
        final_dict = {"Photo": photo_info, "Objects": datalist}

        # dump data into JSON file
        print("    Generating JSON file...")
        data_file = os.path.join(data_results_path, filename + ".json")  # same name as original photo
        print("    Writing data into", data_file)
        with open(data_file, "w+") as f:
            json.dump(final_dict, f, indent=4)

        # save saturation stats if necessary
        if generate_saturation_stats:
            print("    Generating saturation stats text file...")
            # build string for saturation statistics
            stats_str = mf.format_saturation_stats(stats_list, photo, photo_x, photo_y)
            # write saturation stats into .txt file
            text_file = os.path.join(saturation_maps_path, filename + ".txt")  # same name as original photo
            print("    Writing saturation statistics into", text_file)
            with open(text_file, "w+") as f:
                f.write(stats_str)

            # save saturation map
            # saturation map must be saved as png to avoid JPEG color compression
            print("    Generating saturation map...")
            map_file = os.path.join(saturation_maps_path, filename + ".png")  # same name as original photo
            print("    Saving saturation map into", map_file)
            sat_img = Image.fromarray(saturation_map)
            sat_img.save(map_file)


# ----------------------------
# object detection execution
# ----------------------------
print("---STARTING OBJECT DETECTION---")
start1 = time.time()
data = run_detection(photos_dir, results_dir)
end1 = time.time()
print("---DONE---")
elapsed_time_detection = end1 - start1
elapsed_time_detection_min = elapsed_time_detection / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(elapsed_time_detection, elapsed_time_detection_min))

print("\n\n---GENERATING ADDITIONAL DATA FILES---")
start2 = time.time()
generate_data_files(data)
end2 = time.time()
print("---DONE---")
elapsed_time_files = end2 - start2
elapsed_time_files_min = elapsed_time_files / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(elapsed_time_files, elapsed_time_files_min))
