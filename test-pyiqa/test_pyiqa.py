import pyiqa
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyiqa.utils
import torch
import os
import time
import json
import pprint
import gc
from corrupt_images import corrupt_image
from utils import log, bcolors
import json
import argparse
import yaml

def run_metric_varied(path, metrics, output_path, device, file_types=()):
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
        print(score)

    with open(output_path, 'w') as json_file:
        json.dump(img_dict, json_file, indent=4)

    pprint.pprint(img_dict)
    print(f'Process took {time.time() - start} seconds')

def verify_metrics(metrics):
    for metric in metrics:
        if metric not in pyiqa.list_models():
            print(f"{metric} not in pyiqa's models")
            return False
    return True

def generate_scores(base_path, img_paths, metric, device):
    log(f"Running metric: {metric}", bcolors.OKBLUE)
    # Is this enough for it to run on CUDA when available. Do I also need .cuda()?
    model = pyiqa.create_metric(metric, device=device)
    score_dict = {}
    for img_path in img_paths:
        # Unsqueezing is done to get a 4D tensor
        img_tensor = pyiqa.utils.img_util.imread2tensor(os.path.join(base_path, img_path)).unsqueeze(0)
        if verify_tensor(img_tensor=img_tensor, img=img_path):
            score_dict[img_path] = model(img_tensor).item()
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
        log(f"WARNING: image {img} has bad tensor shape: {img_tensor.shape}. Skipping imgage", 
            bcolors.WARNING)
        return False
    return True

def to_tensor(nd_arr):
    """
    Transforms nd_arr to pytorch tensor. In addition, transposes the array
    and transforms values from [0, 255] -> [0, 1]
    """
    transposed = np.transpose(nd_arr, (2, 0, 1))
    return (torch.tensor(transposed).unsqueeze(0) / 255)

def construct_image_dict(corrupt_image_dict, img_name):
    dict = {}
    for corruption, severity_dicts in corrupt_image_dict.items():
        for severity in severity_dicts.keys():
            dict[f"{img_name}_{corruption}_{severity}.png"] = {}
    return dict

def label_images(image_dict):
    for filename, scores in image_dict.items():
        image = Image.open(f"output/corrupted/{filename}")
        new_image = Image.new('RGB', (image.width, image.height + 50*len(scores.keys())), (0, 0, 0))
        new_image.paste(image, (0, 0))
        font = ImageFont.load_default(size=36)
        draw = ImageDraw.Draw(new_image)
        position = (0, image.height + 10)
        text = '\n'.join(f'{key}: {value}' for key, value in scores.items())
        draw.text(position, text, (255, 255, 255), font=font) 
        new_image.save(f"output/corrupted/{filename}")

def noisy_images(corruption_types, metrics, path, img_name, results_path):
    log("Running noisy_images", bcolors.OKBLUE)
    corrupt_image_dict = corrupt_image(path, img_name, corruptions=corruption_types)
    # Keys are names of images with a corruption type and a severity type.
    # Values consist of dictionaries with key-value pairs of type: (metric: score)
    image_dict = construct_image_dict(corrupt_image_dict, img_name)
    for metric in metrics:
        log(f"Running metric: {metric}", bcolors.OKBLUE)
        model = pyiqa.create_metric(metric, device="cpu")
        for corruption, corruption_dict in corrupt_image_dict.items():
            for severity, image_nd in corruption_dict.items():
                img_tensor = to_tensor(image_nd)
                score = model(img_tensor).item()
                rounded_score = round(score, 4)
                key = f"{img_name}_{corruption}_{severity}.png"
                image_dict[key][metric] = rounded_score
                print(f"{corruption}, {severity}, {metric} -> score: {rounded_score}")
    label_images(image_dict)
    with open(results_path, 'w') as json_file:
        json.dump(image_dict, json_file, indent=4)

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    #gc.collect()
    #torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="pyiqa runnner")
    parser.add_argument("-c", "--config", type=str, help="path to where the config .yaml file is located")
    args = parser.parse_args()
    if not args.config:
        raise argparse.ArgumentTypeError("Directory path must be specified")
    else:
        config = load_config(args.config)
        
    if not config['add_noise']:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("device:", device)
        run_metric_varied(path=config['img_dir'],
                metrics=config['metrics'],
                file_types=tuple(config['file_types']),
                output_path=config['output_dir'],
                device=device)
    else:
        noisy_images(corruption_types=config['corruption_types'], 
                     metrics=config['metrics'],
                     path=config['img_path'],
                     img_name=config['img_name'],
                     results_path=config['results_path'])

if __name__ == '__main__':
    main()