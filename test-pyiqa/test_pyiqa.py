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
import argparse
from corrupt_images import corrupt_image
from utils import bcolors
import json

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
        if verify_tensor(img_tensor=img_tensor, img_path=img_path):
            score_dict[img_path] = model(img_tensor).item()
    return score_dict

def generate_score(img_np, metric, model, device="cpu"):
    print(f"\n{bcolors.OKBLUE}Running metric: {metric}{bcolors.ENDC}\n")
    #model = pyiqa.create_metric(metric, device=device)
    img_tensor = to_tensor(img_np)
    if verify_tensor(img_tensor=img_tensor):
        return model(img_tensor).item()
    else:
        raise ValueError(f"Tensor: {img_np.shape} has bad shape. Crashing")

def verify_tensor(img_tensor, img=""):
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

def to_tensor(nd_arr):
    """
    Transforms nd_arr to pytorch tensor. In addition, transposes the array
    and transforms values from [0, 255] -> [0, 1]
    """
    transposed = np.transpose(nd_arr, (2, 0, 1))
    return (torch.from_numpy(transposed).unsqueeze(0) / 255)    

def construct_image_dict(corrupt_image_dict):
    dict = {}
    for corruption, severity_dicts in corrupt_image_dict.items():
        for severity in severity_dicts.keys():
            dict[f"1.png_{corruption}_{severity}.png"] = {}
    return dict

def label_images(image_dict):
    for filename, scores in image_dict.items():
        image = Image.open(f"corrupted/{filename}")
        new_image = Image.new('RGB', (image.width, image.height + 50*len(scores.keys())), (0, 0, 0))
        new_image.paste(image, (0, 0))
        font = ImageFont.load_default(size=36)
        draw = ImageDraw.Draw(new_image)
        position = (0, image.height + 10)
        text = '\n'.join(f'{key}: {value}' for key, value in scores.items())
        draw.text(position, text, (255, 255, 255), font=font) 
        new_image.save(f"corrupted/{filename}")

def noisy_images():
    print("Running noisy_images")
    corruption_types = ["gaussian_noise", "motion_blur", "defocus_blur"]
    metrics = ["brisque_matlab", "niqe_matlab", "hyperiqa", "ilniqe"]
    corrupt_image_dict = corrupt_image("../../../sample_imgs/1.png", "1.png", corruptions=corruption_types)
    # Keys are names of images with a corruption type and a severity type.
    # Values consist of dictionaries with key-value pairs of type: (metric: score)
    image_dict = construct_image_dict(corrupt_image_dict)
    for metric in metrics:
        print(f"\n{bcolors.OKBLUE}Running metric: {metric}{bcolors.ENDC}\n")
        model = pyiqa.create_metric(metric, device="cpu")
        for corruption, corruption_dict in corrupt_image_dict.items():
            for severity, image_nd in corruption_dict.items():
                img_tensor = to_tensor(image_nd)
                score = model(img_tensor).item()
                key = f"1.png_{corruption}_{severity}.png"
                image_dict[key][metric] = score
                print(f"{corruption}, {severity}, {metric} -> score: {score}")
    label_images(image_dict)

def debug():
    metrics = ["hyperiqa"]
    image_path = "../../../sample_imgs/1.png"
    img_tensor = pyiqa.utils.img_util.imread2tensor(image_path).unsqueeze(0)
    print(f"img_tensor.dtype: {img_tensor.dtype}")
    #generate_score()

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
    #main()
    #debug()
    noisy_images()