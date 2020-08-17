import cv2 as cv
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from PIL import Image
import tensorflow as tf
import numpy as np
import tarfile
import requests
import os


# loads TensorFlow model from locally saved files, downloads them if they not yet exist
def load_tensorflow_model(model_name):
    print("Proceeding to load model: {}...".format(model_name))

    # file strings
    base_url = "http://download.tensorflow.org/models/object_detection/"
    download_dir = "downloaded_models"
    model_dir = os.path.join(download_dir, model_name, "saved_model")
    model_graph_path = model_dir + "/saved_model.pb"
    model_tar_file = model_name + ".tar.gz"
    model_tar_filepath = os.path.join(download_dir, model_tar_file)

    print("Checking if graph file exists: {}...".format(model_graph_path))
    # check if graph file exists
    if not os.path.exists(model_graph_path):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # retrieve tar file if it not exists
        print("Model graph file does not exist, checking if model TAR file exists...")
        if not os.path.exists(model_tar_file):
            print("    Downloading model TAR file from {}...".format(base_url + model_tar_file))
            r = requests.get(base_url + model_tar_file)
            with open(model_tar_filepath, "wb") as f:
                f.write(r.content)
        else:
            print("Model TAR file exists, skipping download...")
        # extract graph file from tar file
        print("Extracting graph from {}...".format(model_tar_filepath))
        tar_file = tarfile.open(model_tar_filepath)
        tar_file.extractall(download_dir)
        # close file
        tar_file.close()

    print("Loading model...")
    model = tf.saved_model.load(model_dir)
    model = model.signatures["serving_default"]
    print("Done.")

    return model


# loads YOLO model with OpenCV, downloads YOLO weights file if it doesn't exist
def load_yolo_model():
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    download_dir = "downloaded_models/YOLOv3"
    weights_file = download_dir + "/yolov3.weights"

    cfg_file = "data/YOLO/yolov3.cfg"

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
    coco_names = "data/YOLO/coco.names"
    labels_list = open(coco_names).read().strip().split("\n")
    category_index = {i: {"id": i, "name": labels_list[i]} for i in range(len(labels_list))}

    return category_index


# function for object detection with TensorFlow model
def run_image_inference(model, image, apply_nms=False, nms_threshold=0.3):
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

    # apply non maximum suppression if specified
    if apply_nms:
        boxes = output_dict["detection_boxes"]
        scores = output_dict["detection_scores"]
        nms_indices = tf.image.non_max_suppression(boxes, scores, len(boxes), iou_threshold=nms_threshold).numpy()
        output_dict["detection_boxes"] = output_dict["detection_boxes"][nms_indices]
        output_dict["detection_scores"] = output_dict["detection_scores"][nms_indices]
        output_dict["detection_classes"] = output_dict["detection_classes"][nms_indices]

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
def detect_with_yolo(yolo_model, image, suppression_threshold=0.3):
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
    indexes = cv.dnn.NMSBoxes(nms_boxes, confidence_scores, 0, suppression_threshold)
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
def get_inference_image(image_path, output_dict, category_index, line_thickness=1, score_threshold=0.5):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    with Image.open(image_path) as opened_image:
        image_np = np.array(opened_image)
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


def absolute_bounding_box(box, x, y):
    y_min, x_min, y_max, x_max = box
    return [y_min * y, x_min * x, y_max * y, x_max * x]


# increments every element in numpy 2D array on places where bounding box rectangle should be located - "draws" the box
def draw_box(np_array, box_dimensions):
    # round dimensions and convert to int
    min_x = int(round(box_dimensions["min x"]))
    max_x = int(round(box_dimensions["max x"]))
    min_y = int(round(box_dimensions["min y"]))
    max_y = int(round(box_dimensions["max y"]))
    # check if out of bounds
    img_y, img_x = np_array.shape
    if max_x > img_x:
        max_x = img_x
    if max_y > img_y:
        max_y = img_y
    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0

    # calculate used width and height from rounded values
    width = max_x - min_x
    height = max_y - min_y

    # increment array elements where bounding box is located
    box = np.ones((height, width))
    np_array[min_y:max_y, min_x:max_x] += box


# counts number occurrences in range 0-10+
def calculate_saturation_stats(image_array):
    stats_list = []

    # for values 0-9
    for i in range(10):
        num_values = np.count_nonzero(image_array == i)
        stats_list.append(num_values)

    # last element in list indicates 10+ objects, so condition must be >= 10
    num_values = np.count_nonzero(image_array >= 10)
    stats_list.append(num_values)

    return stats_list


# normalizes incremented values from draw_box() function into range 0-255
def normalize_grayscale(np_array):
    # multiply every value with 255/10
    np_array = (np_array * 25.5).astype(int)
    # set upper threshold
    np_array[np_array > 255] = 255
    # reverse colors (more objects == darker area)
    np_array = (255 - np_array).astype(np.uint8)

    return np_array


# formats stats list into string for saving into text file
def format_saturation_stats(stats_list, image_name, img_width, img_height):
    all_pixels = img_width * img_height
    percentages = [(value / all_pixels) * 100 for value in stats_list]

    # initial stats string
    stats_str = "SATURATION STATISTICS FOR " + image_name

    # full image occupation percentage is equal to sum of all values with value for 0 objects excluded
    occupied_image = sum(percentages[1:])
    # add row for image occupation percentage
    stats_str = "{}\nPercentage of image occupied: {} %".format(stats_str, round(occupied_image, 4))

    # add rows for 1-9 objects
    stats_str += "\nNumber of objects:"
    for i in range(1, len(percentages) - 1):
        stats_str = "{}\n  {}: {} %".format(stats_str, i, round(percentages[i], 4))

    # for last element, format it as "10+"
    stats_str = "{}\n10+: {} %".format(stats_str, round(percentages[-1], 4))

    return stats_str
