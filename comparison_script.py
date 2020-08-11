import py_functions.comparison_functions as cf
import time
import os

models = {
    "f_rcnn": "faster_rcnn_nas_coco_2018_01_28",
    "r_fcn": "rfcn_resnet101_coco_2018_01_28",
    "ssd": "ssd_mobilenet_v1_coco_2018_01_28",
    "yolo": "YOLOv3"
}
res_dir = "Results/2020-08-08 Piran_en"
node_color1 = "yellow"
node_color2 = "red"
iou_threshold = 0.7


def generate_heatmaps_and_node_graphs(models_dict, results_dir,  color1, color2, iou_thresh=0.7, node_graphs="pgv"):
    thresh_str = "threshold={}".format(iou_thresh)
    # dir for heat maps
    dir_heatmaps = os.path.join(results_dir, "comparison", thresh_str, "heat_maps")
    if not os.path.exists(dir_heatmaps):
        os.makedirs(dir_heatmaps)
    # dir for node graphs with pair counts as weights
    dir_graphs = os.path.join(results_dir, "comparison", thresh_str, "node_graphs", "weights=pair_counts")
    if not os.path.exists(dir_graphs):
        os.makedirs(dir_graphs)
    # dir for node graphs with Jaccard index as weights
    dir_graphs_jaccard = os.path.join(results_dir, "comparison", thresh_str, "node_graphs", "weights=Jaccard_index")
    if not os.path.exists(dir_graphs_jaccard):
        os.makedirs(dir_graphs_jaccard)

    for model1 in models_dict.values():
        for model2 in models_dict.values():
            if os.path.exists(dir_heatmaps + "/{} with {}.png".format(model2, model1)):
                continue
            # set names for files and value of used IOU threshold
            # IOU threshold is 1, if we compare model with itself
            if model1 != model2:
                save_file = "{} with {}.png".format(model1, model2)
                used_iou = iou_thresh
            else:
                save_file = "{} with itself.png".format(model1)
                used_iou = 1

            # where heat map will be saved
            heat_map_path = os.path.join(dir_heatmaps, save_file)
            # where node graph for pair counts will be saved
            graph_path = os.path.join(dir_graphs, save_file)
            # where node graph for Jaccard indices will be saved
            graph_path_jaccard = os.path.join(dir_graphs_jaccard, save_file)

            # compare models
            print("Comparing {} with {}".format(model1, model2))
            start = time.time()
            comparison_data = cf.compare_two_models(results_dir, model1, model2, used_iou)
            end = time.time()
            print("Time elapsed: {} s".format(round(end - start, 4)))

            # get object counts (for node graph with Jaccard indices)
            object_counts = (comparison_data["object_count1"], comparison_data["object_count2"])
            # get all location pairs
            pairs = cf.get_common_location_pairs(comparison_data)
            # get name lists from pairs
            names1, names2 = cf.name_lists_from_pairs(pairs)
            # reverse names for second model, so y labels on heat map will increase alphabetically from bottom up
            names2 = names2[::-1]

            # generate heat map matrix
            print("    Saving heat map...")
            start = time.time()
            heat_map = cf.get_heat_map(pairs, names1, names2)
            # save image file for heat map matrix
            cf.save_heat_map(heat_map, model1, model2, names1, names2, heat_map_path)
            end = time.time()
            print("    Time elapsed: {} s".format(round(end - start, 4)))

            # save image files for node graphs
            print("    Saving node graphs...")
            start = time.time()
            assert node_graphs in ["pgv", "nx"], "Invalid parameter for node graph type"
            if node_graphs == "pgv":
                cf.nodes_graph_pgv(pairs, color1, color2, graph_path)
                cf.nodes_graph_pgv(pairs, color1, color2, graph_path_jaccard,
                                   use_jaccard=True, object_counts=object_counts)
            elif node_graphs == "nx":
                cf.nodes_graph_nx(pairs, color1, color2, graph_path)
                cf.nodes_graph_nx(pairs, color1, color2, graph_path_jaccard,
                                  use_jaccard=True, object_counts=object_counts)
            end = time.time()
            print("    Time elapsed: {} s".format(round(end - start, 4)))


print("---BEGINNING HEAT MAP AND NODE GRAPHS GENERATION---")
t1 = time.time()
generate_heatmaps_and_node_graphs(models, res_dir, node_color1, node_color2, iou_thresh=iou_threshold)
t2 = time.time()
print("---DONE---")
print("Total time elapsed: {} s".format(round(t2 - t1, 4)))
