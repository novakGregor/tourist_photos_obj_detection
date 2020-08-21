import py_functions.comparison_functions as cf
import time
import os


def generate_heatmaps_and_node_graphs(models_dict, results_dir,  color1, color2,
                                      iou_thresh=0.7):
    # list with model combinations that don't have any common detected objects
    # is returned by function
    no_match = []

    # string for folder name that indicates used iou threshold
    thresh_str = "iou_threshold={}".format(iou_thresh)

    # ==================================================================
    # variables for directory paths
    # ==================================================================

    # ------------------------------------------------------------------
    # dir for heat maps
    # ------------------------------------------------------------------
    dir_heatmaps = os.path.join(results_dir, "comparison", thresh_str, "heat_maps")
    if not os.path.exists(dir_heatmaps):
        os.makedirs(dir_heatmaps)

    # ------------------------------------------------------------------
    # paths for node graphs variations
    # ------------------------------------------------------------------
    # base dir str for node graphs
    dir_graphs_base_str = os.path.join(results_dir, "comparison", thresh_str, "node_graphs",
                                       "widths={}", "width_threshold={}")
    # dir paths for all variations of node graphs
    # NOTE: only currently used weight thresholds are 3 when using object counts and 0.01 when using Jaccard index
    dir_graphs = dir_graphs_base_str.format("pair_counts", 0)
    dir_graphs_thresh = dir_graphs_base_str.format("pair_counts", 3)
    dir_graphs_jaccard = dir_graphs_base_str.format("Jaccard_index", 0)
    dir_graphs_jaccard_thresh = dir_graphs_base_str.format("Jaccard_index", 0.01)
    # create dirs that don't exist
    for dir_path in [dir_graphs, dir_graphs_thresh, dir_graphs, dir_graphs_jaccard, dir_graphs_jaccard_thresh]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # ------------------------------------------------------------------
    # paths for table variations
    # ------------------------------------------------------------------
    # base table dir str
    tables_base_str = os.path.join(results_dir, "comparison", thresh_str, "tables",
                                   "ordered_by={}", "{}_threshold={}")
    # paths for all used variations
    counts_tables = tables_base_str.format("pair_counts", "count", 0)
    counts_tables_thresh = tables_base_str.format("pair_counts", "count", 3)
    jaccard_tables = tables_base_str.format("Jaccard_index", "index", 0)
    jaccard_tables_thresh = tables_base_str.format("Jaccard_index", "index", 0.01)
    # create dirs that don't exist
    for dir_path in [counts_tables, counts_tables_thresh, jaccard_tables, jaccard_tables_thresh]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # ==================================================================
    # double for loop with model comparison
    # ==================================================================
    for model1 in models_dict.values():
        for model2 in models_dict.values():
            # skip models if they were already compared
            if os.path.exists(dir_heatmaps + "/{} with {}.png".format(model2, model1)):
                continue
            # compare models with themselves only if iou_thresh == 1
            if model1 == model2 and iou_thresh < 1:
                continue
            elif model1 != model2 and iou_thresh == 1.0:
                continue
            print("Comparing {} with {}".format(model1, model2))

            # ----------------
            # variables for full file paths
            # ----------------
            # filename for saved files
            if model1 == model2:
                save_file = "{}.png".format(model1)
            else:
                save_file = "{} with {}.png".format(model1, model2)
            # path for heat map file location
            heat_map_path = os.path.join(dir_heatmaps, save_file)
            # paths for all graph files
            graph_path = os.path.join(dir_graphs, save_file)
            graph_path_thresh = os.path.join(dir_graphs_thresh, save_file)
            graph_path_jaccard = os.path.join(dir_graphs_jaccard, save_file)
            graph_path_jaccard_thresh = os.path.join(dir_graphs_jaccard_thresh, save_file)
            # paths for all table files
            counts_table_file = os.path.join(counts_tables, save_file)
            counts_table_file_thresh = os.path.join(counts_tables_thresh, save_file)
            jaccard_table_file = os.path.join(jaccard_tables, save_file)
            jaccard_table_file_thresh = os.path.join(jaccard_tables_thresh, save_file)

            # comparison execution
            comparison_data = cf.compare_two_models(results_dir, model1, model2, iou_thresh)

            # get object counts
            object_counts = (comparison_data["object_count1"], comparison_data["object_count2"])

            # get all location pairs
            pairs = cf.get_common_location_pairs(comparison_data)
            # skip current models if there was no match and update no_match list
            if not pairs:
                no_match_str = "{} -> {}\n".format(model1, model2)
                no_match.append(no_match_str)
                continue

            # ------------------------------------------------------------------
            # generate heat map matrix
            # ------------------------------------------------------------------
            # get name lists from pairs
            names1, names2 = cf.name_lists_from_pairs(pairs)
            # reverse names for second model, so y labels on heat map will increase alphabetically from bottom up
            names2 = names2[::-1]
            print("    Saving heat map...")
            heat_map = cf.get_heat_map(pairs, names1, names2)
            # save image file for heat map matrix
            cf.save_heat_map(heat_map, object_counts, model1, model2, names1, names2, heat_map_path)

            # ------------------------------------------------------------------
            # generate table images and node graphs
            # ------------------------------------------------------------------
            print("    Saving tables and node graphs...")
            models_tuple = (model1, model2)
            # ----------------
            # without thresholds
            # ----------------
            # tables
            cf.plot_pair_table(pairs, counts_table_file, models_tuple, object_counts)
            cf.plot_pair_table(pairs, jaccard_table_file, models_tuple, object_counts,
                               jaccard_primary=True)
            # node graphs
            cf.nodes_graph_pgv(pairs, color1, color2, graph_path)
            cf.nodes_graph_pgv(pairs, color1, color2, graph_path_jaccard,
                               use_jaccard=True, object_counts=object_counts)
            # ----------------
            # with thresholds
            # execute in try/except -> if failed, no matches when using threshold
            # ----------------
            # strings for no_match list
            no_match_count_thresh = ""
            no_match_jaccard_thresh = ""

            # width = object counts
            try:
                cf.plot_pair_table(pairs, counts_table_file_thresh, models_tuple, object_counts, count_thresh=3)
                cf.nodes_graph_pgv(pairs, color1, color2, graph_path_thresh,
                                   weight_thresh=3)
            except ValueError:
                print("SAVING TABLE FAILED: Table with count threshold empty")
                no_match_count_thresh = "{} -> {}, count threshold=3\n".format(model1, model2)

            # width = Jaccard index
            try:
                cf.nodes_graph_pgv(pairs, color1, color2, graph_path_jaccard_thresh,
                                   use_jaccard=True, jaccard_thresh=0.01, object_counts=object_counts)
                cf.plot_pair_table(pairs, jaccard_table_file_thresh, models_tuple, object_counts,
                                   jaccard_primary=True, jaccard_thresh=0.01)
            except ValueError:
                print("SAVING TABLE FAILED: Table with Jaccard index threshold empty")
                no_match_jaccard_thresh = "{} -> {}, Jaccard index threshold=0.1\n".format(model1, model2)

            # if strings are not empty, append them to no_match list
            for string in [no_match_count_thresh, no_match_jaccard_thresh]:
                if string:
                    no_match.append(string)

    return no_match


models = {
    "f_rcnn-openimages": "faster_rcnn_inception_resnet_v2_atrous_oidv4",
    "f_rcnn-slow": "faster_rcnn_nas_coco",
    "f_rcnn": "faster_rcnn_resnet50_coco",
    "r_fcn": "rfcn_resnet101_coco",
    "ssd": "ssd_mobilenet_v1_coco",
    "ssd2": "ssd_mobilenet_v1_fpn_coco",
    "ssd-openimages": "ssd_mobilenetv2_oidv4",
    "yolo3": "YOLOv3"
}

res_dir = "Results/2020-08-17 Piran_en"
node_color1 = "yellow"
node_color2 = "red"
iou_threshold = 0.5


print("---BEGINNING HEAT MAP AND NODE GRAPHS GENERATION---")
start = time.time()

# generate files for all iou thresholds
while iou_threshold <= 1.0:
    print("\n***Current iou threshold: {}***".format(iou_threshold))
    t1 = time.time()
    zero_matches = generate_heatmaps_and_node_graphs(models, res_dir, node_color1, node_color2,
                                                     iou_thresh=iou_threshold)
    # list is not empty -> there were models with no matches,
    # so create no matches info file
    if zero_matches:
        curr_thresh_dir = "iou_threshold={}".format(iou_threshold)
        zero_matches_file = os.path.join(res_dir, "comparison", curr_thresh_dir, "zero_matches.txt")
        with open(zero_matches_file, "w+") as f:
            for item in zero_matches:
                f.write(item)
    t2 = time.time()
    print("Time elapsed: {} s".format(round(t2 - t1, 4)))

    # increase threshold by 0.05
    iou_threshold += 0.05
    iou_threshold = round(iou_threshold, 2)

end = time.time()
print("---DONE---")
print("Total time elapsed: {} s".format(round(end - start, 4)))
