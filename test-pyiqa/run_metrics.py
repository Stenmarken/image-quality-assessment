import pyiqa
import numpy as np
import pyiqa.utils
import torch
import os
import time
import json
import pprint
import gc
from utils import log, bcolors, load_yaml
import argparse


def run_metric(path, metrics, output_path, device, file_types=()):
    """
    Calculates metric for all files with file_types in location path.

    Can work with images of varying dimensions
    """
    start = time.time()
    if not verify_metrics(metrics):
        return False

    img_paths = [f for f in os.listdir(path) if f.lower().endswith(file_types)]
    img_dict = {}

    for metric in metrics:
        score = generate_scores(path, img_paths, metric, device)
        img_dict[metric] = score

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as json_file:
        json.dump(img_dict, json_file, indent=4)

    pprint.pprint(img_dict)
    print(f"Process took {time.time() - start} seconds")


def verify_metrics(metrics):
    for metric in metrics:
        if metric not in pyiqa.list_models():
            print(f"{metric} not in pyiqa's models")
            return False
    return True


def generate_scores(base_path, img_paths, metric, device):
    start = time.time()
    log(f"Running metric: {metric}", bcolors.OKBLUE)
    # Is this enough for it to run on CUDA when available. Do I also need .cuda()?
    model = pyiqa.create_metric(metric, device=device)
    score_dict = {}
    for img_path in img_paths:
        log(f"Running {img_path}", bcolors.OKBLUE)
        # Unsqueezing is done to get a 4D tensor
        img_tensor = pyiqa.utils.img_util.imread2tensor(
            os.path.join(base_path, img_path)
        ).unsqueeze(0)
        if verify_tensor(img_tensor=img_tensor, img=img_path):
            score_dict[img_path] = model(img_tensor).item()
    print(f"Running metric took {time.time() - start} seconds")
    return score_dict


def generate_score(img_np, metric, device="cpu"):
    log(f"Running metric: {metric}", bcolors.OKBLUE)
    model = pyiqa.create_metric(metric, device=device)
    img_tensor = to_tensor(img_np)
    if verify_tensor(img_tensor=img_tensor):
        return model(img_tensor).item()
    else:
        raise ValueError(f"Tensor: {img_np.shape} has bad shape. Crashing")


def verify_tensor(img_tensor, img=""):
    if img_tensor.shape[1] not in [1, 3]:
        log(
            f"WARNING: image {img} has bad tensor shape: {img_tensor.shape}. Skipping imgage",
            bcolors.WARNING,
        )
        return False
    return True


def to_tensor(nd_arr):
    """
    Transforms nd_arr to pytorch tensor. In addition, transposes the array
    and transforms values from [0, 255] -> [0, 1]
    """
    transposed = np.transpose(nd_arr, (2, 0, 1))
    return torch.tensor(transposed).unsqueeze(0) / 255


def construct_image_dict(corrupt_image_dict, img_name):
    dict = {}
    for corruption, severity_dicts in corrupt_image_dict.items():
        for severity in severity_dicts.keys():
            dict[f"{img_name}_{corruption}_{severity}.png"] = {}
    return dict


def main():
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="pyiqa runnner")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to where the config .yaml file is located",
    )
    args = parser.parse_args()
    if not args.config:
        raise argparse.ArgumentTypeError("Directory path must be specified")
    else:
        config = load_yaml(args.config)
        print(f"args.config: {args.config}")
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("device:", device)
        print("config", config)
        run_metric(
            path=config["img_dir"],
            metrics=config["metrics"],
            file_types=tuple(config["file_types"]),
            output_path=config["output_path"],
            device=device,
        )


if __name__ == "__main__":
    main()
