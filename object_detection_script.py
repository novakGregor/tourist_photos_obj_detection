from object_detection.utils import label_map_util
from datetime import datetime
import py_functions.model_functions as mf
import py_functions.detection_functions as df
import time
import os

# dict with name strings for models from TensorFlow object detection zoo
models = {
    "ssd": "ssd_mobilenet_v1_coco_2018_01_28",
    "f_rcnn": "faster_rcnn_nas_coco_2018_01_28",
    "m_rcnn": "mask_rcnn_inception_v2_coco_2018_01_28",
    "r_fcn": "rfcn_resnet101_coco_2018_01_28",
    "yolo": "YOLOv3"
}

# ----------------------------
# variables for running object detection
# ----------------------------

# path to file from which labels for TensorFlow model are read
PATH_TO_LABELS = "Data/mscoco_label_map.pbtxt"

# value for score threshold
score_threshold = 0.5

# name string for used model
used_model = models["ssd"]

# actual model for object detection
if used_model != "YOLOv3":
    model = mf.load_tensorflow_model(used_model)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
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
data = df.run_detection(model, used_model, category_index, photos_dir_path, results_dir,
                        use_yolo=use_yolo, score_threshold=score_threshold)
end1 = time.time()
print("---DONE---")
elapsed_time_detection = end1 - start1
elapsed_time_detection_min = elapsed_time_detection / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(elapsed_time_detection, elapsed_time_detection_min))

print("\n\n---GENERATING ADDITIONAL DATA FILES---")
start2 = time.time()
df.generate_data_files(data, category_index, generate_saturation_stats=True)
df.generate_csv(data, category_index, used_model, results_dir)  # full csv
df.generate_csv_by_folder(data, category_index, used_model)  # csv files in each subdir
df.generate_json(data, category_index, used_model, results_dir)  # full json
df.generate_json_by_folder(data, category_index, used_model)  # json files in each subdir
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
