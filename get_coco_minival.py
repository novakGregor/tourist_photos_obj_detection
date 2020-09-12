from object_detection.utils import label_map_util
from collections import defaultdict
import shutil
import json
import os


def create_minival(val_path, minival_path, minival_ids_path):
    base_filename = "COCO_val2014_{}.jpg"

    with open(minival_ids_path) as f:
        ids = f.readlines()

    for id_num in ids:
        photo_id = id_num.strip()
        # number of id digits in photos' filenames is always 12
        if len(photo_id) < 12:
            # each filename has exactly 12 id digits, first digits are always zeros, so add
            # as many zeros as there are missing for id to have 12 digits
            photo_id = "0" * (12 - len(photo_id)) + photo_id
        print("Current id: {}; {}/{}".format(photo_id, ids.index(id_num), len(ids)))
        photo_filename = base_filename.format(photo_id)
        photo_path = os.path.join(val_path, photo_filename)
        shutil.copy(photo_path, minival_path)


# reformats coco dataset's JSON structure into JSON structure, that is primarily used in this project
def reformat_coco_annotations(coco_json, basedir, categories):
    full_list = []
    objects_dict = defaultdict(list)

    print("Loading JSON into python dictionary...")
    with open(coco_json) as f:
        json_data = json.load(f)

    # relevant data from json
    annotations = json_data["annotations"]
    img_infos = json_data["images"]

    print("Reading objects...")
    for obj in annotations:
        img_id = obj["image_id"]
        # for ground truth, every score is considered 100%
        score = 1.0
        # object class/category and its proper name
        class_id = obj["category_id"]
        class_name = categories[class_id]["name"]

        # reformat bounding box into appropriate dictionary
        min_x, min_y, b_width, b_height = obj["bbox"]
        max_x = min_x + b_width
        max_y = min_y + b_height
        bbox = {"width": b_width,
                "height": b_height,
                "min x": min_x,
                "max x": max_x,
                "min y": min_y,
                "max y": max_y}

        # append current obj dict into full objects dict
        obj_dict = {"Class id": class_id,
                    "Class name": class_name,
                    "Score": score,
                    "Bounding box": bbox}
        objects_dict[img_id].append(obj_dict)

    print("Creating image dicts...")

    # rewrite data into final structure
    for img in img_infos:
        filename = os.path.join(basedir, img["file_name"])
        img_id = img["id"]
        width = img["width"]
        height = img["height"]
        objects = objects_dict[img_id]

        img_dict = {"Photo": filename, "Width": width, "Height": height, "Objects": objects}
        full_list.append(img_dict)

    return full_list


def save_coco_minival_json(val_json, minival_ids_path, save_path):
    with open(minival_ids_path) as f:
        ids = [line.strip() for line in f.readlines()]

    minival_list = []
    ids2 = []
    print("Creating minival JSON list...")
    for img in val_json:
        img_filename = os.path.basename(img["Photo"])
        img_id = img_filename.split("_")[-1]
        img_id = img_id.split(".")[0]
        img_id = img_id.lstrip("0")
        if img_id in ids:
            ids2.append(img_id)
            minival_list.append(img)
    minival_list = sorted(minival_list, key=lambda x: x["Photo"])
    print("Saving list into JSON...")
    with open(save_path, "w+") as f:
        json.dump(minival_list, f, indent=4)


# bools for determining which parts of code to execute
copy_minival = False
create_minival_json = True

# path to file from which minival ids are read
minival_ids = "data/mscoco_minival_ids.txt"

# copy minival photos from val dataset to specified path for minival dataset
if copy_minival:
    coco_val_path = "Photos/mscoco/val2014"
    coco_minival_path = "Photos/mscoco/minival2014"
    if not os.path.exists(coco_minival_path):
        os.makedirs(coco_minival_path)
    create_minival(coco_val_path, coco_minival_path, minival_ids)

# save restructured json file for minival dataset
if create_minival_json:
    json_path = "data/coco_annotations/instances_val2014.json"
    new_json = "data/mscoco_minival2014_ground_truth.json"
    tf_coco_labels = "data/mscoco_label_map.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(tf_coco_labels, use_display_name=True)
    minival_json_list = reformat_coco_annotations(json_path, "Photos", category_index)
    save_coco_minival_json(minival_json_list, minival_ids, new_json)
