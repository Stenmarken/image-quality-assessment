import pyiqa
from PIL import Image
import numpy as np
import pyiqa.utils
import torch
import os
import time
import pprint
import json

def run_metric_varied(path, metrics, output_path, file_types=()):
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
        score = generate_scores(path, img_paths, metric)
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

def generate_scores(base_path, img_paths, metric):
    print(f"Running metric: {metric}")
    model = pyiqa.create_metric(metric)
    score_dict = {}
    for img_path in img_paths:
        # Unsqueezing is done to get a 4D tensor
        img_tensor = pyiqa.utils.img_util.imread2tensor(os.path.join(base_path, img_path)).unsqueeze(0)
        score_dict[img_path] = model(img_tensor).item()
    return score_dict

def run_metric(path, metric, output_path, file_types=()):
    """
    Calculates metric for all files with file_types in location path.

    Requires that the input files are of equal dimensions.
    """
    start = time.time()
    img_paths = [f for f in os.listdir(path) if f.lower().endswith(file_types)]
    images = [Image.open(os.path.join(path, img_path)) for img_path in img_paths]
    
    img_tensors = map(lambda img: pyiqa.utils.img_util.imread2tensor(img), images)
    model = pyiqa.create_metric(metric, device=device)
    score = model(torch.stack(list(img_tensors)))
    with open(output_path, 'w') as f:
        for i in range(len(img_paths)):
            f.write(f'{metric} score for {img_paths[i]}: {score[i]}\n')
            print(f'{metric} score for {img_paths[i]}: {score[i]}')
    
    print(f'Process took {time.time() - start} seconds')

path = "../../../sample_imgs"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# clipiqa+_vitL14_512 model doesn't seem to work. Can't load it in. It loops forever.
print("device:", device)
metrics = ['arniqa-clive', 'arniqa-csiq', 'arniqa-flive', 
           'arniqa-kadid', 'arniqa-live', 'arniqa-spaq', 
           'arniqa-tid', 'brisque_matlab', 'clipiqa', 'clipiqa+', 
           'clipiqa+_rn50_512', #'clipiqa+',
           'cnniqa', 'dbcnn'] #, 'entropy', 'hyperiqa']


#run_metric(path=path, metric="brisque_matlab", file_types=(".png"), output_path="run_metric.txt")
run_metric_varied(path=path, metrics=metrics, file_types=(".jpg", ".jpeg"), output_path="output.json")

#run_metric(path="../../../sample_imgs", metric="brisque", file_types=(".png"), 
           #output_path="run_metric_brisque.txt")
#run_metric_varied(path="../../../sample_imgs", metric="brisque", file_types=(".png"), 
           #output_path="run_metric_unconforming.txt")