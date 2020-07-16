from object_detection.utils import label_map_util
from datetime import datetime
import model_functions as mf
import detection_functions as df
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
# object detection execution
# ----------------------------
print("---STARTING OBJECT DETECTION---")
start1 = time.time()
data = df.run_detection(model, category_index, photos_dir, results_dir)
end1 = time.time()
print("---DONE---")
elapsed_time_detection = end1 - start1
elapsed_time_detection_min = elapsed_time_detection / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(elapsed_time_detection, elapsed_time_detection_min))

print("\n\n---GENERATING ADDITIONAL DATA FILES---")
start2 = time.time()
df.generate_data_files(data, category_index, generate_saturation_stats=True)
df.generate_csv(data, category_index, results_dir)  # full csv
df.generate_csv_by_folder(data, category_index)  # csv files in each subdir
df.generate_json(data, category_index, results_dir)  # full json
df.generate_json_by_folder(data, category_index)  # json files in each subdir
end2 = time.time()
print("---DONE---")
elapsed_time_files = end2 - start2
elapsed_time_files_min = elapsed_time_files / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(elapsed_time_files, elapsed_time_files_min))
