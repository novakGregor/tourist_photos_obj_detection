from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2 as cv
import tarfile
import requests
import os


# loads TensorFlow model from locally saved files, downloads them if they not yet exist
def load_tensorflow_model(model_name):
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


# loads YOLO model with OpenCV, downloads YOLO weights file if it doesn't exist
def load_yolo_model():
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    download_dir = "downloaded_models/YOLOv3"
    weights_file = download_dir + "/yolov3.weights"

    cfg_file = "Data/YOLO/yolov3.cfg"

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print("Checking if file for YOLOv3 weights exists...")
    if not os.path.exists(weights_file):
        print("    Downloading weights file (this will take some time as website is slow)...")
        r = requests.get(weights_url)
        with open(weights_file, "wb") as f:
            f.write(r.content)
        print("    Done.")
    else:
        print("    Skipping file as it already exists")

    # net, used as detection model
    net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)

    return net


# returns labels dict for YOLO, which can be used in the same way as category index for TensorFlow models
def get_yolo_labels():
    coco_names = "Data/YOLO/coco.names"
    labels_list = open(coco_names).read().strip().split("\n")
    category_index = {i: {"id": i, "name": labels_list[i]} for i in range(len(labels_list))}

    return category_index


# function which does actual detection with TensorFlow model
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


# function that runs object detection with YOLO model
# returns similar output dict as TensorFlow function, so it can be used with same visualization function
def detect_with_yolo(yolo_model, image, score_threshold=0.5, suppression_threshold=0.3):
    # load our input image and grab its spatial dimensions
    height, width = image.shape[0], image.shape[1]

    # determine only the *output* layer names that we need from YOLO
    layer_names = yolo_model.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    layer_outputs = yolo_model.forward(layer_names)

    # min x, min y, max x, max y coordinates for each detected bounding box
    # later used for visualization of bounding boxes
    min_max_boxes = []
    # min x, min y, width, height for each detected bounding box
    # used for performing non maximum suppression
    nms_boxes = []
    confidence_scores = []
    classes = []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence score of the current object detection
            results = detection[5:]
            class_id = np.argmax(results)
            confidence_score = results[class_id]

            # perform score threshold check
            if confidence_score < score_threshold:
                continue

            # scale bounding box coordinates to image size
            # yolo returns center x y coordinates + width and height of bounding box
            box = detection[0:4] * np.array([width, height, width, height])
            (center_x, center_y, box_width, box_height) = box.astype("int")

            # calculate min and max coordinates of a bounding box
            min_x = int(round(center_x - (box_width / 2)))
            max_x = int(round(center_x + (box_width / 2)))
            min_y = int(round(center_y - (box_height / 2)))
            max_y = int(round(center_y + (box_height / 2)))

            # create list with min and max coordinates, divided with width and height
            # such list can be used for visualisation in the same way as with tensorflow models
            min_max_box = np.array([min_y / height, min_x / width, max_y / height, max_x / width])

            # append results to lists
            min_max_boxes.append(min_max_box)
            nms_boxes.append([min_x, min_y, int(box_width), int(box_height)])
            confidence_scores.append(float(confidence_score))
            classes.append(class_id)

    # apply non maximum suppression to suppress weak, overlapping bounding boxes
    indexes = cv.dnn.NMSBoxes(nms_boxes, confidence_scores, score_threshold, suppression_threshold)
    indexes = np.array(indexes)
    output_dict = {"detection_boxes": [], "detection_classes": [], "detection_scores": []}
    for i in indexes.flatten():
        output_dict["detection_boxes"].append(min_max_boxes[i])
        output_dict["detection_classes"].append(classes[i])
        output_dict["detection_scores"].append(confidence_scores[i])

    # convert lists to numpy arrays so they can be used with TensorFlow visualization function
    output_dict["detection_boxes"] = np.array(output_dict["detection_boxes"])
    output_dict["detection_classes"] = np.array(output_dict["detection_classes"])
    output_dict["detection_scores"] = np.array(output_dict["detection_scores"])

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
