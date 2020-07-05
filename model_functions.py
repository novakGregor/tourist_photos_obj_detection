from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from PIL import Image
import tensorflow as tf
import numpy as np
import tarfile
import pathlib
import requests
import os


# loads model from URL, downloaded files are not saved on drive
def load_model(model_name):
    base_url = "http://download.tensorflow.org/models/object_detection/"
    model_file = model_name + ".tar.gz"
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"
    print(str(model_dir))

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures["serving_default"]

    return model


# loads model from locally saved files, downloads them if they not yet exist
def load_local_model(model_name):
    print("Proceeding to load local model...")

    # file strings
    base_url = "http://download.tensorflow.org/models/object_detection/"
    download_dir = "downloaded_models"
    model_dir = os.path.join(download_dir, model_name, "saved_model")
    saved_model_path = model_dir + "/saved_model.pb"
    model_file = model_name + ".tar.gz"
    model_filepath = os.path.join(download_dir, model_file)

    print("Checking if files exist for", model_name)
    # retrieve file if it not yet exists, else skip current model
    if not os.path.exists(model_filepath):
        print("    Downloading file:", model_file)
        r = requests.get(base_url + model_file)
        with open(model_filepath, "wb") as f:
            f.write(r.content)
    else:
        print("    Skipping file as it already exists:", model_filepath)

    # open downloaded file and extract the frozen graph
    print("Checking if extraction needed for", model_name)
    if not os.path.exists(saved_model_path):
        print("    Extracting graph from", model_filepath)
        tar_file = tarfile.open(model_filepath)
        tar_file.extractall(download_dir)
        # close file
        tar_file.close()
    else:
        print("    Skipping extraction as model already extracted:", saved_model_path)

    model = tf.saved_model.load(model_dir)
    model = model.signatures["serving_default"]

    return model


# function which does actual detection
def run_image_inference(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors
    # Convert to numpy arrays, and take index [0] to remove the batch dimension
    # We"re only interested in the first num_detections
    num_detections = int(output_dict.pop("num_detections"))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict["num_detections"] = num_detections

    # detection_classes should be ints.
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)

    # Handle models with masks:
    if "detection_masks" in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict["detection_masks"], output_dict["detection_boxes"], image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return output_dict


# visually applies bounding boxes and other detected data on the photo
# wrapper function for object_detection's visualize_boxes_and_labels_on_image_array() function
def get_inference_image(image_path, output_dict, category_index, line_thickness=1, score_threshold=.5):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict["detection_boxes"],
        output_dict["detection_classes"],
        output_dict["detection_scores"],
        category_index,
        instance_masks=output_dict.get("detection_masks_reframed", None),
        use_normalized_coordinates=True,
        line_thickness=line_thickness,
        min_score_thresh=score_threshold)

    return image_np


# calculates absolute coordinates and width and height of a bounding box
# returns them as dictionary
def calculate_box_dimensions(box, x, y):
    y_min, x_min, y_max, x_max = box
    y_min1 = y_min * y
    y_max1 = y_max * y
    x_min1 = x_min * x
    x_max1 = x_max * x
    width = x_max1 - x_min1
    height = y_max1 - y_min1

    box_dict = {
        "width": width,
        "height": height,
        "min x": x_min1,
        "max x": x_max1,
        "min y": y_min1,
        "max y": y_max1
    }
    return box_dict


# draws filled box on given numpy image array and simultaneously updates pixel statistics
def draw_box(np_image, box_dimensions, stats_list):
    # round dimensions and convert to int
    min_x = int(round(box_dimensions["min x"]))
    max_x = int(round(box_dimensions["max x"]))
    min_y = int(round(box_dimensions["min y"]))
    max_y = int(round(box_dimensions["max y"]))
    # check if out of bounds
    img_x = np_image.shape[1]
    img_y = np_image.shape[0]
    if max_x >= img_x:
        max_x = img_x - 1
    if max_y >= img_y:
        max_y = img_y - 1

    # draw rectangle
    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            # get old and new color values
            old_color = np_image[i][j][0]
            new_color = old_color - 25
            # indexes for accessing values in stats_list
            # need to be rounded because most values will not be dividable by 25
            old_index = int(round((255 - old_color) / 25))
            new_index = int(round((255 - new_color) / 25))
            # fix stats values
            # final value on index 0 will be equal to negative value of all pixels that don't remain white
            stats_list[old_index] -= 1
            stats_list[new_index] += 1

            if new_color <= 5:  # 255 - (25 * 10) = 5 -> almost black
                new_color = 0  # 10+ objects
            np_image[i][j] = [new_color, new_color, new_color]


# formats stats list into string for saving into text file
def format_saturation_stats(stats_list, image_name, img_width, img_height):
    all_pixels = img_width * img_height
    percentages = [(value / all_pixels) * 100 for value in stats_list]

    # initial stats string
    stats_str = "SATURATION STATISTICS FOR " + image_name

    # first value in list can be negative number
    # because value for 0 objects is subtracted for each pixel with different color
    occupied_image = abs(percentages[0])
    # add row for image occupation percentage
    stats_str = "{}\nPercentage of image occupied: {} %".format(stats_str, round(occupied_image, 4))

    # add rows for 1-9 objects
    stats_str += "\nNumber of objects:"
    for i in range(1, len(percentages) - 1):
        stats_str = "{}\n  {}: {} %".format(stats_str, i, round(percentages[i], 4))

    # for last element, format it as "10+"
    stats_str = "{}\n10+: {} %".format(stats_str, round(percentages[-1], 4))

    return stats_str
