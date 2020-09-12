from collections import defaultdict
import py_functions.comparison_functions as cf
import numpy as np
import json
import csv
import os


# ==============================================================
# first 5 functions can be used for filtering tab files for Booking.com photos
# according to what objects do various models detect on those photos
# ==============================================================


# computes all detected objects frequencies for json data file
# and returns it as dictionary
def get_object_counts(model_json_file):
    with open(model_json_file) as f:
        model_data = json.load(f)
    object_counts = defaultdict(int)
    for photo in model_data:
        objects = photo["Objects"]
        for obj in objects:
            obj_name = obj["Class name"]
            object_counts[obj_name] += 1
    return object_counts


# gets list with dictionaries of object counts for multiple models
# and sums up all frequencies
# doesn't consider potential intersection in object counts
def merge_object_counts(counts_dicts_list):
    object_counts = counts_dicts_list[0]
    for obj_counts in counts_dicts_list[1:]:
        for obj in obj_counts:
            object_counts[obj] += obj_counts[obj]
    return object_counts


# saves dictionary with counts into tsv (tab) file
def save_counts_to_tab(object_counts, tab_file_path):
    counts_tuples = sorted(list(object_counts.items()), key=lambda x: -x[1])
    with open(tab_file_path, "w+") as f:
        for i in range(len(counts_tuples)):
            obj_count_list = [i+1] + list(counts_tuples[i])
            f.write("\t".join(map(str, obj_count_list)) + "\n")


# reads tsv (tab) file with counts and saves counts into list
def read_counts_from_tab(tab_file_path):
    names = []
    with open(tab_file_path) as f:
        for row in csv.reader(f, dialect="excel-tab"):
            names.append(row[1])
    return names


# computes object counts for all specified models, sums them together and saves it into count file
def save_all_model_counts(models_list, results_basedir, tab_file_path):
    counts_dicts_list = []
    json_base_filename = "all_photos_data - {}.json"
    for model in models_list:
        json_file_path = os.path.join(results_basedir, model, json_base_filename.format(model))
        obj_counts = get_object_counts(json_file_path)
        counts_dicts_list.append(obj_counts)
    joined_counts = merge_object_counts(counts_dicts_list)
    save_counts_to_tab(joined_counts, tab_file_path)


# joins detection data from multiple models into one dictionary
# data is not yet merged
def join_models(models_list, results_path):
    # values of joined dict are tuples with two dictionaries
    # item on tuple's index 0 is dictionary for photo's info (name, width, height)
    # item on tuple's index 1 is dictionary for objects (keys=models, values=object list)
    joined_dict = defaultdict(lambda: ({}, {}))

    print("Building json...")
    base_json_filename = "all_photos_data - {}.json"
    for model in models_list:
        json_path = os.path.join(results_path, model, base_json_filename.format(model))
        with open(json_path) as f:
            results_data = json.load(f)
        for photo_dict in results_data:
            photo = photo_dict["Photo"]
            photo_width = photo_dict["Width"]
            photo_height = photo_dict["Height"]
            joined_dict[photo][0]["photo"] = photo
            joined_dict[photo][0]["width"] = photo_width
            joined_dict[photo][0]["height"] = photo_height

            joined_dict[photo][1][model] = photo_dict["Objects"]

    return joined_dict


# ==============================================================
# following functions are used for merging models' detection data
# into detection system with multiple models
# ==============================================================


# combines objects' data from multiple models into lists that contain data for every model
# therefore every object's data is now "multidimensional"
def combine_objects_data(joined_dict, iou_thresh):
    # list which will have combined data for each photo and will be returned by function
    full_photos_list = []

    print("Combining objects...")
    photo_num = 1
    for photo in joined_dict:
        photo_num += 1

        photo_info, objects_dict = joined_dict[photo]
        combined_objects = []
        used_objects = set()

        models = list(objects_dict.keys())
        for i in range(len(models)):
            model1 = models[i]

            # retrieve model1's objects and
            # sort them by score value with descending order,
            # so objects with higher score will be treated first
            model1_objects = sorted(objects_dict[model1], key=lambda x: x["Score"], reverse=True)
            # iterate though model1's objects
            for obj1 in model1_objects:
                # indicator for determining if current object was already used
                obj1_indicator = (obj1["Class name"], tuple(obj1["Bounding box"].items()))
                # skip current object if it was already used
                if obj1_indicator in used_objects:
                    continue
                # add current object to list with used objects
                used_objects.add(obj1_indicator)

                val_list = list(obj1.values())
                all_models_val_list = []
                # for each object's value, create combined value list for all models
                for val_i in range(len(val_list)):
                    # each value list has None values at first, its length == number of models
                    models_val_list = [None] * len(models)
                    # add value of current obj1 on index for current model1
                    models_val_list[i] = val_list[val_i]
                    all_models_val_list.append(models_val_list)

                # create dictionary for combined object info
                id_list, name_list, score_list, bbox_list = all_models_val_list
                combined_obj = {
                    "id": id_list,
                    "name": name_list,
                    "score": score_list,
                    "bbox": bbox_list
                }

                # compare model1's object with other models' objects
                bbox1 = obj1["Bounding box"]
                for model2 in objects_dict:
                    # don't compare model with itself
                    if model1 == model2:
                        continue

                    # retrieve model2's objects and
                    # sort them by score value with descending order,
                    # so objects with higher score will be treated first
                    model2_objects = sorted(objects_dict[model2], key=lambda x: x["Score"], reverse=True)
                    # compare each obj2 with obj1
                    for obj2 in model2_objects:
                        # create indicator for obj2
                        obj2_indicator = (obj2["Class name"], tuple(obj2["Bounding box"].items()))
                        # retrieve bounding box and calculate IOU with obj1
                        bbox2 = obj2["Bounding box"]
                        iou = cf.get_iou(bbox1, bbox2)
                        # skip current object if IOU not big enough OR object was already used
                        if iou < iou_thresh or obj2_indicator in used_objects:
                            continue

                        # add obj2 to used objects
                        used_objects.add(obj2_indicator)
                        # retrieve values for obj2 and write them into dict with combined values
                        id2, name2, score2, bbox2 = obj2.values()
                        model2_i = models.index(model2)
                        combined_obj["id"][model2_i] = id2
                        combined_obj["name"][model2_i] = name2
                        combined_obj["score"][model2_i] = score2
                        combined_obj["bbox"][model2_i] = bbox2
                        # because match was already found, there's no need to compare other model2's objects
                        break

                # add dict with combined values to list with all combined objects for current photo
                combined_objects.append(combined_obj)

        # retrieve info about current photo
        photo_name, photo_width, photo_height = photo_info.values()
        # write all values into dict and append it into full list for all photos
        photo_dict = {
            "Photo": photo_name,
            "Width": photo_width,
            "Height": photo_height,
            "Objects": combined_objects
        }
        full_photos_list.append(photo_dict)

    return full_photos_list


# executes model merging so that it creates new objects from "multidimensional" objects' data
# which are now standard "onedimensional" objects
# then saves merged data into standard json file
def merge_combined_objects_data(combined_data, save_path):
    full_merged_data_list = []
    print("Merging objects...")
    for photo in combined_data:
        photo_name, width, height, objects = photo.values()
        combined_objects = []
        for obj in objects:
            # retrieve combined data from obj dict
            id_list, name_list, score_list, bbox_list = obj.values()
            # join ids and names into one string
            joined_id = "-".join(map(str, id_list))
            joined_name = "-".join(map(str, name_list))

            # for average score calculation, ignore scores that are None
            # if we treated None scores as 0, average score would be smaller, but not necessarily more correct
            true_scores = [score for score in score_list if score is not None]
            avg_score = np.mean(true_scores)

            # retrieve index of max score
            # and use it for retrieving bounding box for object with max score
            max_score_idx = score_list.index(max(true_scores))
            # bounding box for object with max score will be used as combined object's bounding box
            used_bbox = bbox_list[max_score_idx]

            combined_obj = {
                "Class id": joined_id,
                "Class name": joined_name,
                "Score": avg_score,
                "Bounding box": used_bbox
            }
            combined_objects.append(combined_obj)

        photo_dict = {
            "Photo": photo_name,
            "Width": width,
            "Height": height,
            "Objects": combined_objects
        }
        full_merged_data_list.append(photo_dict)

    print("Saving JSON file...")
    with open(save_path, "w+") as f:
        json.dump(full_merged_data_list, f, indent=4)
