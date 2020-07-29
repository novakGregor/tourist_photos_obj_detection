import py_functions.comparison_functions as cf
import time
import os

models = {
    "ssd": "ssd_mobilenet_v1_coco_2018_01_28",
    "f_rcnn": "faster_rcnn_nas_coco_2018_01_28",
    "r_fcn": "rfcn_resnet101_coco_2018_01_28",
    "yolo": "YOLOv3"
}
res_dir = "Results/2020-07-27 Piran_en"
iou_threshold = 0.7


def generate_all_heatmaps(models_dict, result_dir, iou_thresh=0.7):
    for model1 in models_dict:
        model_name1 = models[model1]

        # dir with heat maps
        save_dir = os.path.join(result_dir, model_name1, "comparison_results", "heat_maps")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for model2 in models:
            model_name2 = models[model2]

            # set name for heat map file and used IOU threshold
            # IOU threshold is 1, if we compare model with itself
            if model1 != model2:
                heat_map_file = "{} with {}.png".format(model_name1, model_name2)
                used_iou = iou_thresh
            else:
                heat_map_file = "{} with itself.png".format(model_name1)
                used_iou = 1

            # where heat map will be saved
            heat_map_path = os.path.join(save_dir, heat_map_file)

            # compare models
            print("Comparing {} with {}".format(model_name1, model_name2))
            start = time.time()
            comparison_data = cf.compare_two_models(result_dir, model_name1, model_name2, used_iou)
            end = time.time()
            print("Time elapsed: {} s".format(end - start))
            # get names for first model
            names1 = comparison_data["all_names1"]
            # reverse names for second model, so y labels on heat map will increase alphabetically from bottom up
            names2 = comparison_data["all_names2"][::-1]
            # generate heat map matrix
            print("    Saving heat map...")
            start = time.time()
            heat_map = cf.get_heat_map(comparison_data, names1, names2)
            # save image file for heat map matrix
            cf.save_heat_map(heat_map, model_name1, model_name2, names1, names2, heat_map_path)
            end = time.time()
            print("    Time elapsed: {} s".format(end - start))


print("---BEGINNING HEAT MAP GENERATION---")
t1 = time.time()
generate_all_heatmaps(models, res_dir, iou_thresh=iou_threshold)
t2 = time.time()
print("---DONE---")
print("Total time elapsed: {} s".format(t2 - t1))
