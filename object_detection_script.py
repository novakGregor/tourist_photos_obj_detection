import py_functions.model_functions as mf
import py_functions.detection_functions as df
from object_detection.utils import label_map_util
from datetime import datetime
import time
import os

# dict for models from TensorFlow object detection zoo
models = {
    "ssd": ("ssd_mobilenet_v1_coco_2018_01_28", "coco"),
    "ssd2": ("ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03", "coco"),
    "ssd-openimages": ("ssd_mobilenet_v2_oid_v4_2018_12_12", "openimages"),
    "f_rcnn": ("faster_rcnn_resnet50_coco_2018_01_28", "coco"),
    "f_rcnn-slow": ("faster_rcnn_nas_coco_2018_01_28", "coco"),
    "f_rcnn-openimages": ("faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12", "openimages"),
    # "m_rcnn": ("mask_rcnn_inception_v2_coco_2018_01_28", "coco"),
    "r_fcn": ("rfcn_resnet101_coco_2018_01_28", "coco"),
    "yolo3": ("YOLOv3", "coco"),
}

# ----------------------------
# variables for running object detection
# ----------------------------

# value for score threshold
score_threshold = 0.5
# name strings for used model and its dataset
used_model, dataset = models["f_rcnn-slow"]
# bool for non maximum suppression application in case of TensorFlow model
tf_apply_nms = True

# prepare object detection model and appropriate category index
if used_model != "YOLOv3":
    model = mf.load_tensorflow_model(used_model)
    # specify appropriate label map path and create category index
    if dataset == "coco":
        tf_labels = "data/mscoco_label_map.pbtxt"
    else:
        tf_labels = "data/oid_v4_label_map.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(tf_labels, use_display_name=True)
    use_yolo = False
else:
    model = mf.load_yolo_model()
    category_index = mf.get_yolo_labels()
    use_yolo = True


# root dir in where directory with photos is located
root_dir = "Photos"

# name of directory from where photos will be used
photos_dir = "Piran_en"

# for differing results folder according to date
today = datetime.today().strftime("%Y-%m-%d")

# full path to directory from where photos will be used
photos_dir_path = os.path.join(root_dir, photos_dir)
# directory where results will be saved
results_dir = "Results/{} {}/{}".format(today, photos_dir, used_model)


# ----------------------------
# object detection execution
# ----------------------------

print("---STARTING OBJECT DETECTION---")
start1 = time.time()
data = df.run_detection(model, used_model, category_index, photos_dir_path, results_dir, apply_nms=tf_apply_nms,
                        use_yolo=use_yolo, score_threshold=score_threshold)
end1 = time.time()
print("---DONE---")
elapsed_time_detection = end1 - start1
elapsed_time_detection_min = elapsed_time_detection / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(elapsed_time_detection, elapsed_time_detection_min))

print("\n\n---GENERATING ADDITIONAL DATA FILES---")
start2 = time.time()
df.generate_data_files(data, category_index, score_threshold=score_threshold, generate_saturation_stats=True)
df.generate_csv(data, category_index, used_model, results_dir, score_thresh=score_threshold)  # full csv
df.generate_csv_by_folder(data, category_index, used_model, score_thresh=score_threshold)  # csv files in each subdir
df.generate_json(data, category_index, used_model, results_dir, score_thresh=score_threshold)  # full json
df.generate_json_by_folder(data, category_index, used_model, score_thresh=score_threshold)  # json files in each subdir
end2 = time.time()
print("---DONE---")
elapsed_time_files = end2 - start2
elapsed_time_files_min = elapsed_time_files / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(elapsed_time_files, elapsed_time_files_min))

print("\n\n---GENERATING BAR CHARTS FOR CONFIDENCE SCORE THRESHOLDS---")
start3 = time.time()
df.save_score_charts(data, used_model, results_dir)  # global charts (for root photos dir)
df.save_score_charts_by_folder(data, used_model)  # local charts (for each subdir)
end3 = time.time()
print("---DONE---")
elapsed_time_charts = end3 - start3
elapsed_time_charts_min = elapsed_time_charts / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(elapsed_time_charts, elapsed_time_charts_min))

time_total = end3 - start1
time_total_min = time_total / 60

print("\nTotal time elapsed: {:.4f} s ({:.4f} m)".format(time_total, time_total_min))
