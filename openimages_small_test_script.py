from object_detection.utils import label_map_util
from collections import defaultdict
from PIL import Image
import shutil
import json
import csv
import os


# joins category index dictionaries with actual names and display names,
# so we can connect ones with the others
def get_full_category_index_dict(name_dict, display_dict):
    full_dict = {}
    for name_id in name_dict:
        name = name_dict[name_id]["name"]
        display_name = display_dict[name_id]["name"]
        full_dict[name] = {"id": name_id, "display_name": display_name}
    return full_dict


# reduces number of images in dataset, copies chosen images to new location
def reduce_dataset_size(skip_factor, old_path, new_path, annotations_json):
    # we will only use photos that contain objects, for determining this, we need annotations dict
    print("Loading annotations dictionary from JSON")
    with open(annotations_json) as f:
        annotations_dict = json.load(f)
    i = 0
    images = os.listdir(old_path)
    len_images = len(images)
    while i < len_images:
        print("Current index: {}/{}".format(i, len_images))
        old_image = os.path.join(old_path, images[i]).replace("\\", "/")  # because keys in dict don't have backslashes
        new_image = os.path.join(new_path, images[i])
        if old_image in annotations_dict:
            shutil.copy(old_image, new_image)
        else:
            print("Skipping file copy for", old_image, "because it doesn't contain any annotated objects")
        i += skip_factor


# reformats openimages annotations into dictionary with values, separated by photos and saves it into JSON file
# this JSON file will then be used to save final JSON file for filtered test dataset,
# with structure that is primarily used in this project
def reformat_openimages_annotations(annotations_path, basedir, categories, save_path):
    # dictionary which will contain annotations for each photo
    annotations_dict = defaultdict(list)
    # base string for image path
    base_image_path = os.path.join(basedir, "test", "{}.jpg")
    print("Reading CSV file...")
    with open(annotations_path) as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader)  # header
        # each row represents one annotated object
        for row in csv_reader:
            # retrieve object's info
            image_path = base_image_path.format(row[0])
            object_name = row[2]
            object_id = categories[object_name]["id"]
            object_display_name = categories[object_name]["display_name"]
            # for ground truth, confidence score is always 100%
            score = 1.0
            # retrieve bounding box coordinates and compute its width and height
            x_min, x_max = float(row[4]), float(row[5])
            y_min, y_max = float(row[6]), float(row[7])
            width = x_max - x_min
            height = y_max - y_min
            # save bounding box values into dictionary
            bbox = {
                "width": width,
                "height": height,
                "min x": x_min,
                "max x": x_max,
                "min y": y_min,
                "max y": y_max
            }
            # create dictionary for current object and add it into annotations dict
            object_dict = {
                "Class id": object_id,
                "Class name": object_display_name,
                "Score": score,
                "Bounding box": bbox}
            annotations_dict[image_path].append(object_dict)

    # add photos information into annotations dictionary
    print("Saving annotations into dictionary...")
    final_dict = {}
    # variables for printing saving statistics to stdout
    photo_num = 1
    dict_len = len(annotations_dict)
    for img in annotations_dict:
        print("Current image: {}; {}/{}".format(img, photo_num, dict_len))
        photo_num += 1
        image = Image.open(img)
        img_width, img_height = image.size
        image.close()
        img_dict = {
            "Photo": image_path,
            "Width": img_width,
            "Height": img_height,
            "Objects": annotations_dict[img]
        }
        final_dict[img] = img_dict

    # save final annotations dict into JSON file
    with open(save_path, "w+") as f:
        json.dump(final_dict, f, indent=4)


# filters full openimages annotations and saves it into file
# with JSON structure, which is primarily used in this project
def filter_json_annotations(full_dict_path, small_test_path, save_path, basedir):
    print("Starting annotations filtering...")
    print("Loading full dict from JSON...")
    with open(full_dict_path) as f:
        full_dict = json.load(f)
    # list with filtered data
    small_test_list = []
    # get all photo names from small test folder
    small_test_photos = os.listdir(small_test_path)

    print("Filtering test dataset annotations...")
    # variables for printing filtering statistics to stdout
    photo_num = 1
    photos_len = len(small_test_photos)
    # for each photo in small test set
    for photo in small_test_photos:
        print("Current photo: {}; {}/{}".format(photo, photo_num, photos_len))
        photo_num += 1

        # read data for current photo and append it into filtered list
        key = os.path.join(basedir, "test", photo)
        new_photo = os.path.join(basedir, "openimages_test_small", photo)
        photo_dict = full_dict[key]
        photo_dict["Photo"] = new_photo
        small_test_list.append(full_dict[key])

    print("Saving filtered data into file", save_path)
    with open(save_path, "w+") as f:
        json.dump(small_test_list, f, indent=4)


# for determining if filtered dataset should be created
copy_small_dataset = False

# strings with used paths
base_photos_dir = "Photos"
openimages_test = "Photos/test"
tf_openimages_labels = "data/oid_v4_label_map.pbtxt"
csv_annotations_path = "data/datasets_annotations/test-annotations-bbox.csv"
full_annotations_json = "data/datasets_annotations/openimages_test.json"
small_test_json = "data/openimages_test_small_ground_truth.json"
small_test = "Photos/openimages_test_small"
if not os.path.exists(small_test):
    os.makedirs(small_test)

# category indices for both display object names and actual object names
display_names = label_map_util.create_category_index_from_labelmap(tf_openimages_labels, use_display_name=True)
true_names = label_map_util.create_category_index_from_labelmap(tf_openimages_labels, use_display_name=False)
# merge them into combined dictionary
category_dict = get_full_category_index_dict(true_names, display_names)

# create JSON files
reformat_openimages_annotations(csv_annotations_path, base_photos_dir, category_dict, full_annotations_json)
filter_json_annotations(full_annotations_json, small_test, small_test_json, base_photos_dir)
# create small test dataset if specified
if copy_small_dataset:
    reduce_dataset_size(10, openimages_test, small_test, full_annotations_json)
