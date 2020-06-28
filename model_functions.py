from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from six.moves import urllib
from PIL import Image
import tensorflow as tf
import numpy as np
import tarfile
import pathlib
import os


def load_model(model_name):
    base_url = "http://download.tensorflow.org/models/object_detection/"
    model_file = model_name + ".tar.gz"
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"
    print(str(model_dir))

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures["serving_default"]

    return model


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
        urllib.request.urlopen(base_url + model_file, model_filepath)
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
    print(model_dir)
    model = model.signatures["serving_default"]

    return model


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


def get_inference_image(model, category_index, image_path, line_thickness=1, score_threshold=.5):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_image_inference(model, image_np)
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
