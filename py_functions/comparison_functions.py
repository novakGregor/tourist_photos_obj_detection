import matplotlib.pyplot as plt
import numpy as np
import json
import os


def get_iou(b_box1, b_box2):
    width1, height1, min_x1, max_x1, min_y1, max_y1 = b_box1.values()
    width2, height2, min_x2, max_x2, min_y2, max_y2 = b_box2.values()

    # determine the coordinates of the intersection rectangle
    x_left = max(min_x1, min_x2)
    y_top = max(min_y1, min_y2)
    x_right = min(max_x1, max_x2)
    y_bottom = min(max_y1, max_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # compute the are of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both bounding boxes
    bb1_area = width1 * height1
    bb2_area = width2 * height2

    # compute the intersection over union
    # union = area1 + area2 - intersection
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def list_intersection(t1, t2):
    return sorted(list(set(t1) & set(t2)))


def compare_two_models(results_dir, first_model, second_model, iou_threshold=0.7):
    # directories for models
    first_model_dir = os.path.join(results_dir, first_model)
    second_model_dir = os.path.join(results_dir, second_model)

    # abort if data doesn't exist
    if not (os.path.exists(first_model_dir) and os.path.exists(second_model_dir)):
        print("Results for at least one model don't exist at specified path!")
        return

    # build paths do data files
    base_filename = "all_photos_data - "
    first_model_data_path = "{}/{}{}.json".format(first_model_dir, base_filename, first_model)
    second_model_data_path = "{}/{}{}.json".format(second_model_dir, base_filename, second_model)

    # load data from JSON files
    with open(first_model_data_path) as f1, open(second_model_data_path) as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # list with comparison result for each photo
    all_photos_comparison = []
    # all appeared names for each model
    all_names1, all_names2 = set(), set()

    # for each photo
    for i in range(len(data1)):
        # names that were detected by both models regardless of location
        common_names = set()
        # objects that were detected by both models at the same location regardless of names
        common_locations = []
        # names that were detected only by particular model
        unique_names1, unique_names2 = set(), set()
        # objects that were detected at particular location only by particular model
        unique_locations1, unique_locations2 = set(), set()

        # sets, used to determine if match was already found for object at particular location
        found_common_locations1, found_common_locations2 = set(), set()

        # objects, detected by first model on current photo
        objects1 = data1[i]["Objects"]
        # objects, detected by second model on current photo
        objects2 = data2[i]["Objects"]
        # names of all detected objects by first model on current photo
        obj_names1 = [obj["Class name"] for obj in objects1]
        # names of all detected objects by second model on current photo
        obj_names2 = [obj["Class name"] for obj in objects2]

        # for each photo, update sets of all appeared names
        all_names1.update(obj_names1)
        all_names2.update(obj_names2)
        # update set for common names with intersection of both lists of all names
        common_names.update(list_intersection(obj_names1, obj_names2))
        # unique names of one model are those that were not detected by the other one
        unique_names1.update([name for name in obj_names1 if name not in obj_names2])
        unique_names2.update([name for name in obj_names2 if name not in obj_names1])

        # for each object, detected by first model
        for obj1 in objects1:
            name1 = obj1["Class name"]
            b_box1 = obj1["Bounding box"]
            # used for determining uniqueness of detected object
            obj1_identifier = (name1, tuple(b_box1.items()))

            # for each object, detected by second model
            for obj2 in objects2:
                name2 = obj2["Class name"]
                b_box2 = obj2["Bounding box"]
                # used for determining uniqueness of detected object
                obj2_identifier = (name2, tuple(b_box2.items()))

                # check if IOU is big enough
                if get_iou(b_box1, b_box2) >= iou_threshold:
                    # IOU is big enough -> objects share their location
                    common_locations.append((name1, name2))

                    # update sets for determining already found matches
                    found_common_locations1.add(obj1_identifier)
                    found_common_locations2.add(obj2_identifier)
                else:
                    # obj1 is unique if current obj2 is last object in list and common object for obj1 was not found
                    if b_box2 == objects2[-1]["Bounding box"] and obj1_identifier not in found_common_locations1:
                        unique_locations1.add(obj1_identifier)
                    # similar for obj2 but the other way around
                    if b_box1 == objects1[-1]["Bounding box"] and obj2_identifier not in found_common_locations2:
                        unique_locations2.add(obj2_identifier)

        # build dict with determined data and add it to full list
        curr_photo_comparison = {
            "Photo": data1[i]["Photo"],
            "common_names": sorted(list(common_names)),
            "unique_names1": sorted(list(unique_names1)),
            "unique_names2": sorted(list(unique_names2)),
            "common_locations": common_locations,
            "unique_locations1": list(unique_locations1),
            "unique_locations2": list(unique_locations2)
        }
        all_photos_comparison.append(curr_photo_comparison)

    # return all relevant data
    return_dict = {
        "model1": first_model,
        "model2": second_model,
        "all_names1": sorted(list(all_names1)),
        "all_names2": sorted(list(all_names2)),
        "comparison": all_photos_comparison
    }
    return return_dict


def get_common_location_pairs(comparison_dict):
    pairs = []
    comparison_list = comparison_dict["comparison"]
    for photo in comparison_list:
        pairs.extend(photo["common_locations"])
    return pairs


def get_heat_map(comparison_dict, names1, names2):
    common_location_pairs = get_common_location_pairs(comparison_dict)

    # 2D array (matrix)
    # x -> names for model1
    # y -> names for model2
    heat_map = np.zeros((len(names2), len(names1)))

    for pair in common_location_pairs:
        x = names1.index(pair[0])
        y = names2.index(pair[1])
        heat_map[y][x] += 1

    return heat_map


def save_heat_map(heat_map, model1, model2, names1, names2, file_path):
    plt.figure()
    plt.imshow(heat_map)
    ax = plt.gca()

    # set major ticks and their labels
    ax.set_xticks(np.arange(len(names1)))
    ax.set_yticks(np.arange(len(names2)))
    ax.set_xticklabels(names1)
    ax.set_yticklabels(names2)

    # set label size to 5
    ax.tick_params(axis='both', which='major', labelsize=5)

    # rotate the x labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=90, va="center", ha="right", rotation_mode="anchor")

    # set text for each field in matrix
    for i in range(len(names1)):
        for j in range(len(names2)):
            num = int(heat_map[j][i])
            # let zeros be grayed out
            if num == 0:
                clr = "gray"
            else:
                clr = "w"
            ax.text(i, j, num, ha="center", va="center", color=clr, fontsize=3)

    # set title and axis labels
    ax.set_title("Name comparison for objects at the same location")
    ax.set_xlabel(model1)
    ax.set_ylabel(model2)

    # save photo
    plt.tight_layout()
    plt.savefig(file_path, dpi=250)
    plt.close()
