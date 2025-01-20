import json
from utils import log, bcolors
from PIL import Image
import os
import cv2

def combine_metric_data(path):
    """
    Transform data of type: 
        {"img_name_1" : {"ilniqe": 5}, "img_name_2": {"ilniqe": 6}}
    to
        {"ilniqe": {"img_name_1": 5, "img_name_2" : 6}}
    """
    img_dict = read_data(path)
    first_key = list(img_dict.keys())[0] # Could go wrong if empty dict
    metrics = list(img_dict[first_key].keys())
    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = {}

    for img, metrics in img_dict.items():
        for metric, score in metrics.items():
            metric_dict[metric][img] = score

    with open("corrupted/metric_results.json", 'w') as json_file:
        json.dump(metric_dict, json_file, indent=4)

def reverse_search(path, metric, score_range):
    img_dict = read_data(path)
    metric_dict = img_dict[metric]
    min, max = score_range[0], score_range[1]
    return {k: v for k, v in metric_dict.items() if v >= min and v <= max}

def view_scored_images(results_path, image_path, metric, score_range):
    elems = reverse_search(results_path, metric, score_range)
    for img_name, score in elems.items():
        print(os.path.join(image_path, img_name))
        image = cv2.imread(os.path.join(image_path, img_name))
        cv2.imshow('Image', image)
        log(f"{metric}: {img_name} -> {score}", bcolor_type=bcolors.OKBLUE)
        print("Press any key to continue to the next image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def read_data(path):
    with open(path, 'r') as file:
        return json.load(file)
    
#combine_metric_data("corrupted/results.json")s
#view_scored_images("corrupted/metric_results.json", "corrupted", "hyperiqa", (0.3, 0.4))