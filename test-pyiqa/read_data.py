import json
import pprint

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


def read_data(path):
    with open(path, 'r') as file:
        return json.load(file)
    
combine_metric_data("corrupted/results.json")