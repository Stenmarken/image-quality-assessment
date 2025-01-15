import pyiqa
from PIL import Image
import numpy as np
import pyiqa.utils
import torch
import os
import time
import pprint
import json
import gc
import argparse

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
    print(f"\n{bcolors.OKBLUE}Running metric: {metric}{bcolors.ENDC}\n")
    # Is this enough for it to run on CUDA when available. Do I also need .cuda()?
    model = pyiqa.create_metric(metric, device=device)
    score_dict = {}
    for img_path in img_paths:
        # Unsqueezing is done to get a 4D tensor
        img_tensor = pyiqa.utils.img_util.imread2tensor(os.path.join(base_path, img_path)).unsqueeze(0)
        if verify_tensor(img_path, img_tensor):
            score_dict[img_path] = model(img_tensor).item()
    return score_dict

def verify_tensor(img, img_tensor):
    if img_tensor.shape[1] not in [1, 3]:
        print(f"{bcolors.WARNING}WARNING: image {img} has bad tensor shape:\
              {img_tensor.shape}. Skipping image{bcolors.ENDC}")
        return False
    return True

def run_metric(path, metric, output_path, device, file_types=()):
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

def main():
    (path, file_types, output_path) = parse_arguments()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)
    #gc.collect()
    #torch.cuda.empty_cache()

    # clipiqa+_vitL14_512 model doesn't seem to work. Can't load it in. It loops forever.
    # I also get problems running clipiqa+, entropy, hyperiqa, laion_aes, musiq, musiq-ava,
    # musiq-paq2piq, musiq-spaq, paq2piq
    # qalign takes too much memory, perhaps qalign can be used in Google Colab.
    # qalign_4bit and qalign_8bit causes problems with CUDA. 
    # I don't think fid can be used as it requires a set of ground truth images
    # inception_score: TypeError: only integer tensors of a single element can be converted to an index
    metrics =   ['wadiqam_nr', 'ilniqe', 'hyperiqa', 'arniqa-clive', 'arniqa-csiq', 'arniqa-flive', 
                'arniqa-kadid', 'arniqa-live', 'arniqa-spaq', 
                'arniqa-tid', 'brisque_matlab', 'clipiqa', 'clipiqa+', 
                'clipiqa+_rn50_512', 'dbcnn', 'cnniqa', 'liqe', 'liqe_mix',
                'maniqa', 'maniqa-kadid', 'maniqa-pipal', 
                'nima', 'nima-koniq', 'nima-spaq', 'nima-vgg16-ava', 'niqe', 
                'niqe_matlab', 'nrqm', 'pi', 'piqe',
                'topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 
                'topiq_nr-flive', 'topiq_nr-spaq', 'tres', 'tres-flive', 'unique']
    run_metric_varied(path=path,
                  metrics=metrics,
                  file_types=file_types,
                  output_path=output_path,
                  device=device)

def parse_tuple(arg):
    try:
        return tuple(arg.split(","))
    except:
        raise argparse.ArgumentTypeError("File types must be in format '.jpg,.png,.xyz'")

def parse_arguments():
    parser = argparse.ArgumentParser(description="pyiqa runnner")
    parser.add_argument("dir_path", 
                        type=str, help="path to the directory where the images are")
    parser.add_argument("-f", "--filetypes", type=parse_tuple, 
                        help="A tuple of the file types that should be included in the execution. Ex: .jpg,.png")
    parser.add_argument("-o", "--output", type=str, help="Path to the file where the results are stored")
    
    args = parser.parse_args()

    if not args.dir_path:
        raise argparse.ArgumentTypeError("Directory path must be specified")
    if args.filetypes:
        file_types = args.filetypes
    else:
        file_types = (".jpg", ".png", ".jpeg")
    if args.output:
        output_path = args.output
    else:
        output_path = "results.json"

    return (args.dir_path, file_types, output_path)

if __name__ == '__main__':
    main()