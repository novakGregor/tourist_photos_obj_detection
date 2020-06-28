from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from PIL import Image
import model_functions as mf
import tensorflow as tf
import os

# patch tf1 into utils.ops
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile


# list of the strings that is used to add correct label for each box
PATH_TO_LABELS = "label_maps/mscoco_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# names for different models
ssd = "ssd_mobilenet_v1_coco_2017_11_17"
f_rcnn = "faster_rcnn_nas_coco_2018_01_28"
print("Loading model...")

# detection_model = mf.load_model(ssd)
ssd_model = mf.load_local_model(f_rcnn)

# get paths for all testing photos
test_photos_dir = "Photos/testing_photos"
photos_paths = [os.path.join(test_photos_dir, photo) for photo in os.listdir(test_photos_dir)]

print("Starting detection...")
save_dir = "Results/test_results_2020_6_28"  # where photos will be saved
score_thresh = 0.5  # score threshold to be used for detected objects
for photo in os.listdir(test_photos_dir):
    photo_path = os.path.join(test_photos_dir, photo)
    print("Current photo:", photo_path)
    img_np = mf.get_inference_image(ssd_model, category_index, photo_path, score_threshold=score_thresh)
    img = Image.fromarray(img_np)
    if not os.path.exists(save_dir):
        print(save_dir)
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, photo)
    print("    Saving photo on location", save_path)
    img.save(save_path)
