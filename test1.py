from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from datetime import datetime
from PIL import Image
import model_functions as mf
import tensorflow as tf
import numpy as np
import os

# patch tf1 into utils.ops
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile


# list of the strings that is used to add correct label for each box
PATH_TO_LABELS = "label_maps/mscoco_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# dict with name strings for models from TensorFlow object detection zoo
models = {
    "ssd": "ssd_mobilenet_v1_coco_2018_01_28",
    "f_rcnn": "faster_rcnn_nas_coco_2018_01_28"
}

used_model = models["ssd"]
print("Loading model...")
ssd_model = mf.load_local_model(used_model)

# score threshold to be used for detected objects
score_thresh = 0.5

# directory from with photos fro object detection
test_photos_dir = "Photos/testing_photos"
# directory where result photos will be saved
today = datetime.today().strftime("%Y-%m-%d")
save_dir = os.path.join("Results", today + " test_results", used_model)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# get paths for all testing photos
photos_paths = [os.path.join(test_photos_dir, photo) for photo in os.listdir(test_photos_dir)]

print("Starting detection...")
for photo in os.listdir(test_photos_dir):
    photo_path = os.path.join(test_photos_dir, photo)
    print("Current photo:", photo_path)
    photo_np = np.array(Image.open(photo_path))
    # object detection execution
    output_dict = mf.run_image_inference(ssd_model, photo_np)
    # drawing bounding boxes on photo
    photo_np = mf.get_inference_image(photo_path, output_dict, category_index,
                                      line_thickness=2, score_threshold=score_thresh)
    # save resulting image
    img = Image.fromarray(photo_np)
    save_path = os.path.join(save_dir, photo)
    print("    Saving photo on location", save_path)
    img.save(save_path)
