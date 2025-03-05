import json
from utils import log, bcolors, load_json
import os
import cv2


def combine_metric_data(path):
    """
    Transform data of type:
        {"img_name_1" : {"ilniqe": 5}, "img_name_2": {"ilniqe": 6}}
    to
        {"ilniqe": {"img_name_1": 5, "img_name_2" : 6}}
    """
    img_dict = load_json(path)
    first_key = list(img_dict.keys())[0]  # Could go wrong if empty dict
    metrics = list(img_dict[first_key].keys())
    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = {}

    for img, metrics in img_dict.items():
        for metric, score in metrics.items():
            metric_dict[metric][img] = score

    with open("corrupted/metric_results.json", "w") as json_file:
        json.dump(metric_dict, json_file, indent=4)


def reverse_search(path, metric, score_range):
    """
    Return all images for metric in score_range
    """
    img_dict = load_json(path)
    metric_dict = img_dict[metric]
    min, max = score_range[0], score_range[1]
    return {k: v for k, v in metric_dict.items() if v >= min and v <= max}


def view_scored_images(results_path, image_path, metrics, score_range):
    for metric in metrics:
        log(
            f"Viewing images where {metric} score is in [{score_range[0], score_range[1]}]",
            bcolor_type=bcolors.OKBLUE,
        )
        elems = reverse_search(results_path, metric, score_range)
        for img_name, score in elems.items():
            print(os.path.join(image_path, img_name))
            image = cv2.imread(os.path.join(image_path, img_name))
            cv2.imshow("Image", image)
            log(f"{metric}: {img_name} -> {score}", bcolor_type=bcolors.OKBLUE)
            print("Press any key to continue to the next image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# combine_metric_data("corrupted/results.json")
if __name__ == "__main__":
    view_scored_images(
        "corrupted/metric_results.json", "corrupted", ["hyperiqa", "ilniqe"], (0.3, 0.4)
    )
