from collections import defaultdict
import matplotlib.pyplot as plt
import pygraphviz as pgv
import networkx as nx
import numpy as np
import json


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
def jaccard_indices_from_weights(weights, object_counts1, objects_counts2, jaccard_thresh, nodes_mode=True):
    jaccard_indices = {}
    for pair in weights:
        weight = weights[pair]

        # nodes mode -> names in pairs are names of nodes
        # (strings have digit 1 or 2 on last place, indicating which model they belong to)
        if nodes_mode:
            occurrence1 = object_counts1[pair[0][:-1]]
            occurrence2 = objects_counts2[pair[1][:-1]]
        # not nodes mode -> names in pairs are normal names of objects
        else:
            occurrence1 = object_counts1[pair[0]]
            occurrence2 = objects_counts2[pair[1]]
        # union = set1 + set2 - intersection
        union = occurrence1 + occurrence2 - weight
        if union == 0:
            jaccard_index = 0
        else:
            jaccard_index = round(weight / union, 4)
        if jaccard_index > jaccard_thresh:
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
def compare_two_models(model_names, model_json_paths, iou_thresh=0.7):
    model1, model2 = model_names
    model1_json_path, model2_json_path = model_json_paths

    # load data from JSON files
    with open(model1_json_path) as f1, open(model2_json_path) as f2:
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
        # object occurrences, only on current photo
        photo_object_count1, photo_object_count2 = defaultdict(int), defaultdict(int)
        # objects that were detected by both models at the same location regardless of names
        common_locations = []
        # objects, detected by first model on current photo
        objects1 = sorted(data1[i]["Objects"], key=lambda x: -x["Score"])
        # objects, detected by second model on current photo
        objects2 = sorted(data2[i]["Objects"], key=lambda x: -x["Score"])
        # names of all detected objects by first model on current photo
        obj_names1 = [obj["Class name"] for obj in objects1]
        # names of all detected objects by second model on current photo
        obj_names2 = [obj["Class name"] for obj in objects2]

        # update object counts
        for name in obj_names1:
            object_count1[name] += 1
            photo_object_count1[name] += 1
        for name in obj_names2:
            object_count2[name] += 1
            photo_object_count2[name] += 1

        # for each photo, update sets of all appeared names
        all_names1.update(obj_names1)
        all_names2.update(obj_names2)

        used_objects1 = []
        used_objects2 = []

        # for each object, detected by first model
        for obj1 in objects1:
            name1 = obj1["Class name"]
            b_box1 = obj1["Bounding box"]
            obj1_indicator = (name1, b_box1)
            if obj1_indicator in used_objects1:
                continue
            # for each object, detected by second model
            for obj2 in objects2:
                name2 = obj2["Class name"]
                b_box2 = obj2["Bounding box"]
                obj2_indicator = (name2, b_box2)
                if obj2_indicator in used_objects2:
                    continue
                if get_iou(b_box1, b_box2) >= iou_thresh:
                    # IOU is big enough -> objects share their location
                    common_locations.append((name1, name2))
                    used_objects1.append(obj1_indicator)
                    used_objects2.append(obj2_indicator)

        # build dict with determined data and add it to full list
        curr_photo_comparison = {
            "Photo": data1[i]["Photo"],
            "common_locations": common_locations,
            "photo_object_count1": photo_object_count1,
            "photo_object_count2": photo_object_count2
        }
        all_photos_comparison.append(curr_photo_comparison)

    # return all relevant data
    return_dict = {
        "model1": model1,
        "model2": model2,
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
def save_heat_map(heat_map, object_counts, model1, model2, names1, names2, file_path):

    # retrieve dimensions of heat map
    map_height, map_width = heat_map.shape

    # values for setting appropriate figure size for good subplot shapes
    # divided by 10, so image file is not too big
    fig_width = (map_width + 1) / 10
    fig_height = (map_height + 1) / 10
    # computed value for white space between heat map and object count arrays
    spacing = (0.1 / max(fig_width, fig_height))

    # figure and its grid spec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, 2, width_ratios=(1, len(names1)), height_ratios=(len(names2), 1),
                          wspace=spacing, hspace=spacing)

    # subplots for matrix and arrays
    ax0 = fig.add_subplot(gs[0, 1])  # heat map
    ax1 = fig.add_subplot(gs[1, 1], sharex=ax0)  # object counts for model1
    ax2 = fig.add_subplot(gs[0, 0], sharey=ax0)  # object counts for model2

    # array for model1 must be a row
    counts_array1 = np.zeros((1, map_width)).astype(int)
    # array for model2 must be a column
    counts_array2 = np.zeros((map_height, 1)).astype(int)
    # rewrite values from dictionaries into arrays
    obj_counts1, obj_counts2 = object_counts
    for x in range(map_width):
        counts_array1[0][x] = obj_counts1[names1[x]]
    for y in range(map_height):
        counts_array2[y][0] = obj_counts2[names2[y]]

    # get max object count so it can be used as max value when plotting
    max_value = max(set().union(obj_counts1.values(), obj_counts2.values()))

    # display values on subplots
    ax0.imshow(heat_map, vmax=max_value, aspect="auto")
    ax1.imshow(counts_array1, vmax=max_value, aspect="auto")
    ax2.imshow(counts_array2, vmax=max_value, aspect="auto")

    # set major ticks and their labels
    ax1.set_xticks(np.arange(map_width))
    ax1.set_xticklabels(names1)
    ax1.set_yticks(np.arange(1))
    ax1.set_yticklabels(["Object count"])
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax2.set_yticks(np.arange(map_height))
    ax2.set_yticklabels(names2)
    ax2.set_xticks(np.arange(1))
    ax2.set_xticklabels(["Object count"])
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()

    # hide heat map's labels as it is inner subplot
    ax0.label_outer()

    # set label size to 5 for object names
    ax1.tick_params(axis="x", which="major", labelsize=5)
    ax2.tick_params(axis="y", which="major", labelsize=5)
    # set label size to 3.5 for object count
    ax1.tick_params(axis="y", which="major", labelsize=3.5)
    ax2.tick_params(axis="x", which="major", labelsize=3.5)

    # rotate the x labels and set their alignment
    plt.setp(ax1.get_xticklabels(), rotation=90, va="center", ha="right", rotation_mode="anchor")

    # set text annotation for heat map
    for x in range(map_width):
        for y in range(map_height):
            num = int(heat_map[y][x])
            # let zeros be gray and other numbers white
            clr = "gray" if num == 0 else "w"
            ax0.text(x, y, num, ha="center", va="center", color=clr, fontsize=3)
    # do the same thing for both subplots with object counts
    for x in range(map_width):
        num = int(counts_array1[0][x])
        # let zeros be gray and other numbers white
        # but technically, object count can't be 0 -- otherwise it would be already removed at this point
        clr = "gray" if num == 0 else "w"
        ax1.text(x, 0, num, ha="center", va="center", color=clr, fontsize=2.5)
    for y in range(map_height):
        num = int(counts_array2[y][0])
        # let zeros be gray and other numbers white
        # but technically, object count can't be 0 -- otherwise it would be already removed at this point
        clr = "gray" if num == 0 else "w"
        ax2.text(0, y, num, ha="center", va="center", color=clr, fontsize=2.5)

    # set title and axis labels
    plt.suptitle("Name comparison for objects at the same location")
    ax1.set_xlabel(model1)
    ax2.set_ylabel(model2)

    # save photo
    fig.savefig(file_path, dpi=250, bbox_inches="tight")
    plt.close()


# returns dictionary with data for creating node graphs
def dict_for_node_graph(pairs, color1, color2, use_nx, weight_thresh, use_jaccard, jaccard_thresh, object_counts):

    # add ones and twos to names in pairs - for separating nodes according to model
    pairs = [(pair[0] + "1", pair[1] + "2") for pair in pairs]
    # pairs that will be used as edges - remove pairs which have their count/weight smaller than given threshold
    edges = sorted([pair for pair in set(pairs) if pairs.count(pair) >= weight_thresh])
    # save pair counts into dictionary to use them as weights
    weights = {pair: pairs.count(pair) for pair in edges}

    if use_jaccard:
        # jaccard indices will be used as weights instead of object counts
        assert object_counts is not None, "Object counts not given"
        weights = jaccard_indices_from_weights(weights, object_counts[0], object_counts[1], jaccard_thresh)
        # remove edges that don't have big enough Jaccard index (they are not present anymore in weights dict)
        edges = [pair for pair in edges if pair in weights]

    # normalize weights; normalized weights will be used as edge widths
    min_weight = min(weights.values())
    max_weight = max(weights.values())
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
                    use_jaccard=False, jaccard_thresh=0, object_counts=None, weight_thresh=0):
    # retrieve data for graph creation
    graph_dict = dict_for_node_graph(pairs, color1, color2, False, weight_thresh,
                                     use_jaccard, jaccard_thresh, object_counts)
    nodes, color_map, edges, weights, normalized_weights = graph_dict.values()

    # create graph object and specify all fixed pre-determined attributes
    g = pgv.AGraph(dpi=200, pad=0.6)
    g.node_attr["style"] = "filled"
    g.node_attr["shape"] = "circle"
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


# draws node graph with networkx
def nodes_graph_nx(pairs, color1, color2, save_path,
                   use_jaccard=False, object_counts=None, jaccard_thresh=0, weight_thresh=0):
    # retrieve data for graph creation
    graph_dict = dict_for_node_graph(pairs, color1, color2, True, weight_thresh,
                                     use_jaccard, jaccard_thresh, object_counts)
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


def location_pair_tuples(pairs, obj_counts, jaccard_primary, count_thresh, jaccard_thresh):
    tuples = []

    # used pairs are those that have big enough count threshold
    used_pairs = sorted([pair for pair in set(pairs) if pairs.count(pair) >= count_thresh])
    # build dictionary for pair counts
    pair_counts = {pair: pairs.count(pair) for pair in used_pairs}

    # get jaccard indices
    jaccard_indices = jaccard_indices_from_weights(pair_counts, obj_counts[0], obj_counts[1], jaccard_thresh,
                                                   nodes_mode=False)

    # delete pairs that are not used because of jaccard threshold
    used_pairs = [pair for pair in used_pairs if pair in jaccard_indices]
    pair_counts = {pair: pairs.count(pair) for pair in used_pairs}

    # order of used measures is different according to which measure is used as primary
    if jaccard_primary:
        prim_measure = jaccard_indices
        sec_measure = pair_counts
    else:
        prim_measure = pair_counts
        sec_measure = jaccard_indices

    # first group pairs by first object name in pair, with help of defaultdict
    grouped_dict = defaultdict(list)
    for pair in used_pairs:
        # {name1: (count, name2)}
        grouped_dict[pair[0]].append((prim_measure[pair], sec_measure[pair], pair[1]))

    # then order defaultdict's values by counts
    for obj in grouped_dict:
        grouped_dict[obj] = sorted(grouped_dict[obj], key=lambda x: x[0], reverse=True)

    # retrieve items from defaultdict and order them by biggest count
    # lambda x[1][0][0] -> for each dict key's value ([1]), use count from first pair ([0][0])
    sorted_dict_vals = sorted(list(grouped_dict.items()), key=lambda x: x[1][0][0], reverse=True)

    # reformat them into tuples, leaving object counts and Jaccard indices in the middle
    for obj in sorted_dict_vals:
        for pair in obj[1]:
            tuples.append((obj[0], pair[0], pair[1], pair[2]))

    return tuples


def plot_pair_table(pairs, save_path, models, obj_counts, jaccard_primary=False, count_thresh=0, jaccard_thresh=0):
    tuples = location_pair_tuples(pairs, obj_counts, jaccard_primary, count_thresh, jaccard_thresh)

    names1 = [pair_tuple[0] for pair_tuple in tuples]
    names2 = [pair_tuple[-1] for pair_tuple in tuples]

    # can't use .values() because there can be less pairs in tuples because of thresholds
    obj_counts1 = [obj_counts[0][name] for name in names1]
    obj_counts2 = [obj_counts[1][name] for name in names2]

    # primary values are index 1, secondary on index 2 (Jaccard indices or pair counts)
    prim_vals = [pair_tuple[1] for pair_tuple in tuples]
    sec_vals = [pair_tuple[2] for pair_tuple in tuples]

    # strings for showing text in table
    vals = ["{} ({})".format(val1, val2) for val1, val2 in zip(prim_vals, sec_vals)]

    # values that will be shown as table
    table_values = [obj_counts1, names1, vals, names2, obj_counts2]

    # for coloring table cells
    colors1 = []
    colors2 = []
    color_options1 = ["#ffd000", "yellow"]
    color_options2 = ["firebrick", "red"]

    # make colors alternate by groups of first model's object names
    c_idx = 1
    prev_name = ""
    for name in table_values[1]:
        if name != prev_name:
            c_idx = 1 - c_idx
        colors1.append(color_options1[c_idx])
        colors2.append(color_options2[c_idx])
        prev_name = name

    # make main values cells be darker if value is bigger
    # first, create normalization for matplotlib specter
    # otherwise biggest value would be black and smallest white
    # also, normalize values to range 1-300, for better color consistency between different comparisons
    v_min = min(prim_vals)
    v_max = max(prim_vals)
    used_prim_vals = []
    for val in prim_vals:
        norm_val = normalize_to_range(val, v_min, v_max, 1, 300)
        used_prim_vals.append(norm_val)

    v_min = min(used_prim_vals)
    v_max = max(used_prim_vals)
    norm = plt.Normalize(v_min - 150, v_max + 150)
    v_colors = plt.cm.get_cmap("binary")(norm(used_prim_vals)).tolist()

    # create list for table cell's colors
    t_colors = [colors1, colors1, v_colors, colors2, colors2]

    # insert text for row headers
    table_values[0].insert(0, "object counts")
    table_values[1].insert(0, "object names")
    table_values[2].insert(0, "matches")
    table_values[3].insert(0, "object names")
    table_values[4].insert(0, "object counts")

    # insert "white" in each colors list
    # this is done so first cells, used as row headers, are white
    for colors in [colors1, v_colors, colors2]:
        colors.insert(0, "white")

    plt.figure(figsize=(len(tuples)+1, 5))
    the_table = plt.table(cellText=table_values, cellLoc="center", loc="center", cellColours=t_colors)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6)
    plt.axis("off")

    # add info text to image
    if jaccard_primary:
        title_text = "Comparison, ordered by Jaccard indices of matches, grouped by first model's objects"
    else:
        title_text = "Comparison, ordered by number of matches, grouped by first model's objects"
    threshold_text = "Jaccard index threshold = {}" \
                     "\ncount (matches) threshold = {}".format(jaccard_thresh, count_thresh)
    plt.figtext(0.5, 0.8, title_text, ha="center", fontsize=20)
    plt.figtext(0.5, 0.1, threshold_text, ha="center", fontsize=10)
    plt.figtext(0.5, 0.6, models[0], ha="center", fontsize=15)
    plt.figtext(0.5, 0.35, models[1], ha="center", fontsize=15)

    # finally, save plot into image
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
