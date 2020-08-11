from collections import defaultdict
import matplotlib.pyplot as plt
import pygraphviz as pgv
import networkx as nx
import numpy as np
import json
import os


# returns intersection of two lists and sorts it from min to max
def list_intersection(t1, t2):
    return sorted(list(set(t1) & set(t2)))


# normalizes given value in range min_normalized-max_normalized
def normalize_to_range(value, min_data, max_data, min_normalized, max_normalized):
    if max_data - min_data == 0:
        # return in-between value
        return (max_normalized - min_normalized) / 2
    normalized = ((value - min_data) / (max_data - min_data)) * (max_normalized - min_normalized) + min_normalized
    return normalized


# computes jaccard indices from given list of weights and returns them as dictionary
# object's weight is considered as intersection, and max value from models' object counts is considered as union
def jaccard_indices_from_weights(weights, object_counts1, objects_counts2):
    jaccard_indices = {}
    for pair in weights:
        weight = weights[pair]
        occurrence1 = object_counts1[pair[0][:-1]]
        occurrence2 = objects_counts2[pair[1][:-1]]
        max_count = max(occurrence1, occurrence2)
        if max_count == 0:
            jaccard_index = 0
        else:
            jaccard_index = round(weight / max_count, 4)
        jaccard_indices[pair] = jaccard_index
    return jaccard_indices


# returns Intersection Over Union from given bounding boxes
# bounding boxes are given as dictionaries
def get_iou(b_box1, b_box2):
    width1, height1, min_x1, max_x1, min_y1, max_y1 = b_box1.values()
    width2, height2, min_x2, max_x2, min_y2, max_y2 = b_box2.values()

    # determine the coordinates of the intersection rectangle
    x_left = max(min_x1, min_x2)
    y_top = max(min_y1, min_y2)
    x_right = min(max_x1, max_x2)
    y_bottom = min(max_y1, max_y2)

    if x_right < x_left or y_bottom < y_top:
        # there is no intersection
        return 0.0

    # compute the area of intersection
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


# function for model comparison
# returns dictionary with comparison data
def compare_two_models(results_dir, first_model, second_model, iou_thresh=0.7):
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
    # object occurrences for each model
    object_count1, object_count2 = defaultdict(int), defaultdict(int)

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

        # update object counts
        for name in obj_names1:
            object_count1[name] += 1
        for name in obj_names2:
            object_count2[name] += 1

        # for each photo, update sets of all appeared names
        all_names1.update(obj_names1)
        all_names2.update(obj_names2)
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

                if get_iou(b_box1, b_box2) >= iou_thresh:
                    # IOU is big enough -> objects share their location
                    common_locations.append((name1, name2))

                    # update sets for determining already found matches
                    found_common_locations1.add(obj1_identifier)
                    found_common_locations2.add(obj2_identifier)
                else:
                    # obj1 is unique if current obj2 is last object in list and common object for obj1 was not found
                    if b_box2 == objects2[-1]["Bounding box"] and obj1_identifier not in found_common_locations1:
                        unique_locations1.add(obj1_identifier)
                    # similarly for obj2 but the other way around
                    if b_box1 == objects1[-1]["Bounding box"] and obj2_identifier not in found_common_locations2:
                        unique_locations2.add(obj2_identifier)

        # update set for common names with intersection of both lists of all names
        common_names.update(list_intersection(obj_names1, obj_names2))

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
        "object_count1": object_count1,
        "object_count2": object_count2,
        "comparison": all_photos_comparison
    }
    return return_dict


# combines common_location lists for all photos from comparison dictionary into one list and returns it
def get_common_location_pairs(comparison_dict):
    pairs = []
    comparison_list = comparison_dict["comparison"]
    for photo in comparison_list:
        pairs.extend(photo["common_locations"])
    return pairs


# returns two lists with names from location pairs
def name_lists_from_pairs(pairs):
    names1, names2 = set(), set()
    for pair in pairs:
        names1.add(pair[0])
        names2.add(pair[1])
    return sorted(list(names1)), sorted(list(names2))


# builds numpy matrix with pair count, which resembles some sort of heat map
def get_heat_map(pairs, names1, names2):

    # 2D array (matrix)
    # x -> names for model1
    # y -> names for model2
    heat_map = np.zeros((len(names2), len(names1)))

    for pair in set(pairs):
        # find coordinates for x and y
        x = names1.index(pair[0])
        y = names2.index(pair[1])
        # save occurrence at appropriate place in the heat map matrix
        heat_map[y][x] = pairs.count(pair)

    return heat_map


# plots heat map matrix, retrieved with get_heat_map() function and saves it into image file
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


# returns dictionary with data for creating node graphs
def dict_for_node_graph(pairs, color1, color2, use_nx=False, weight_thresh=0,
                        use_jaccard=False, object_counts=None):

    # add ones and twos to names in pairs - for separating nodes according to model
    pairs = [(pair[0] + "1", pair[1] + "2") for pair in pairs]
    # pairs that will be used as edges - remove pairs which have their count/weight smaller than given threshold
    edges = sorted([pair for pair in set(pairs) if pairs.count(pair) >= weight_thresh])
    # save pair counts into dictionary to use them as weights
    weights = {pair: pairs.count(pair) for pair in edges}

    if use_jaccard:
        # jaccard indices will be used as weights instead of object counts
        assert object_counts is not None, "Object counts not given"
        weights = jaccard_indices_from_weights(weights, object_counts[0], object_counts[1])

    # normalize weights; normalized weights will be used as edge widths
    min_weight = weights[min(weights, key=lambda x: weights[x])]
    max_weight = weights[max(weights, key=lambda x: weights[x])]
    normalized_weights = {pair: normalize_to_range(weights[pair], min_weight, max_weight, 0.5, 7)
                          for pair in weights}

    # get two separate node lists for each model
    nodes1 = [edge[0] for edge in edges]
    nodes2 = [edge[1] for edge in edges]
    # join node lists into one set
    nodes = sorted(set().union(nodes1, nodes2))

    # build dictionary for node colors
    color_map = {}
    for node in nodes:
        if node in nodes1:
            color_map[node] = color1
        else:
            color_map[node] = color2
    # networkx must get list/sequence of colors for each node (can't use dictionary)
    if use_nx:
        color_map = color_map.values()

    # return all values as dictionary
    graph_dict = {
        "nodes": nodes,
        "color_map": color_map,
        "edges": edges,
        "weights": weights,
        "normalized_weights": normalized_weights
    }
    return graph_dict


# draws node graph with pygraphviz
def nodes_graph_pgv(pairs, color1, color2, save_path,
                    use_jaccard=False, object_counts=None, weight_thresh=0):
    # retrieve data for graph creation
    graph_dict = dict_for_node_graph(pairs, color1, color2, use_nx=False, weight_thresh=weight_thresh,
                                     use_jaccard=use_jaccard, object_counts=object_counts)
    nodes, color_map, edges, weights, normalized_weights = graph_dict.values()

    # create graph object and specify all fixed pre-determined attributes
    g = pgv.AGraph(dpi=200, pad=0.6)
    g.node_attr["style"] = "filled"
    g.node_attr['shape'] = 'circle'
    g.node_attr["fixedsize"] = True
    g.node_attr["width"] = 0.3
    g.node_attr["fontsize"] = 8
    g.node_attr["fontcolor"] = "#404040"
    g.edge_attr["fontsize"] = 5
    g.edge_attr["len"] = 0.85

    # add all nodes to graph with their corresponding colors
    for node in nodes:
        g.add_node(node, fillcolor=color_map[node], color=None, label=node[:-1])
    # similarly for edges
    for node1, node2 in edges:
        """NOTE: weight can be objects pair occurrence or Jaccard index for objects pair occurrence"""
        # absolute value of weight is used for edge label
        weight = weights[(node1, node2)]
        # normalized weight is used for edge thickness/width
        normalized_weight = normalized_weights[(node1, node2)]
        g.add_edge(node1, node2, color="#808080", label=str(weight), penwidth=normalized_weight)

    # save graph into image file on specified path
    g.layout()
    g.draw(save_path)


# draws node graph with pygraphviz
def nodes_graph_nx(pairs, color1, color2, save_path, use_jaccard=False, object_counts=None, weight_thresh=0):
    # retrieve data for graph creation
    graph_dict = dict_for_node_graph(pairs, color1, color2, weight_thresh=weight_thresh, use_nx=True,
                                     use_jaccard=use_jaccard, object_counts=object_counts)
    nodes, color_map, edges, weights, normalized_weights = graph_dict.values()

    # graph object
    g = nx.Graph()

    # dictionary for node labels used when plotting graph
    node_labels = {}

    # add each node to graph and its label to label dictionary
    for node in nodes:
        node_label = node[:-1]
        g.add_node(node)
        node_labels[node] = node_label

    g.add_edges_from(edges)

    # set matplotlib figure size
    plt.figure(figsize=(10, 10))

    # set networkx layout to spring, with fixed seed
    pos = nx.spring_layout(g, k=0.3, seed=10)

    # draw nodes and their labels
    nx.draw_networkx_nodes(g, pos, node_color=color_map)
    nx.draw_networkx_labels(g, pos, labels=node_labels)

    # draw each edge with its appropriate width
    for edge in g.edges:
        node1 = edge[0]
        node2 = edge[1]
        edge_key = edge
        if edge not in normalized_weights:
            edge_key = (node2, node1)
        nx.draw_networkx_edges(g, pos, edgelist=[edge], width=normalized_weights[edge_key]*3)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weights)

    # save matplotlib figure at given path
    plt.axis("off")
    plt.savefig(save_path, dpi=200)
