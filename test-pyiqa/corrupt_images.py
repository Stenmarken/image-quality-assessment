from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import os
from utils import bcolors, log
import albumentations as A
import albumentations.augmentations.functional as F
import cv2
import random
import json
import yaml
import itertools
import pprint

def generate_droplets(image, share):
    height, width, _ = image.shape
    num_samples = int(share * height*width)
    indices = np.random.choice(height*width, num_samples, replace=False)
    #TODO: Why is it (index%width, index // width). 
    # Shouldn't it be (index // width, index % width)
    coords = [(index % width, index // width) for index in indices]
    return coords

def generate_rain_droplets(path, output_path):
    """
    Seeding randomness doesn't make this function deterministic.
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.ReplayCompose([A.RandomRain(
        slant_range=(-15, -15),
        drop_length=20,
        drop_width=1,
        drop_color=(180, 180, 180),
        blur_value=7,
        brightness_coefficient=0.7,
        p=1.0,
    )])
    transformed = transform(image=image)
    coords = transformed['replay']['transforms'][0]['params']['rain_drops']
    coords_dict = {
        'shape' : image.shape,
        'path' : path,
        'coords' : coords
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coords_dict, f, indent=4)

def get_droplets(droplets_path):
    with open(droplets_path, "r") as f:
        droplets_dict = json.load(f)
        return [tuple(coord) for coord in droplets_dict['coords']]

def func_albumentations(img_path, config):
    #droplets = get_droplets(config['droplets_path'])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    droplets  = generate_droplets(image, config['droplet_share'])
    return F.add_rain(img=image,
               slant=config['slant'],
               drop_length=config['drop_length'],
               drop_width=config['drop_width'],
               drop_color=config['drop_color'],
               blur_value=config['blur_value'],
               brightness_coefficient=config['brightness_coefficient'],
               rain_drops=droplets)

def albumentations_corrupted():
    #drop_lengths = [20, 40, 60]
    drop_lengths = [20, 20, 20]
    #blur_values = [5, 7, 9]
    blur_values = [5, 5, 5]
    #drop_widths = [1, 2, 3]
    drop_widths = [1, 1, 1]

    #brightness_coefficients = [0.7, 0.5, 0.3]
    brightness_coefficients = [0.7, 0.7, 0.7]

    image = cv2.imread('../../../sample_imgs/1.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(1):
        transform = A.ReplayCompose([A.RandomRain(
                    slant_range=(-15, -15),
                    drop_length=drop_lengths[i],
                    drop_width=drop_widths[i],
                    drop_color=(180, 180, 180),
                    blur_value=blur_values[i],
                    brightness_coefficient=brightness_coefficients[i],
                    p=1.0)])
        transformed = transform(image=image)
        image_pil = Image.fromarray(transformed['image'])
        image_pil.show()

def get_corrupted_image(corruption, image_path, severity):
    if not os.path.isfile(image_path):
        raise f"{image_path} is not a valid file. Crashing"
    if corruption in ["glass_blur", "fog"]:
        print(f"{bcolors.WARNING}Corruption type {corruption} is not supported. Returning{bcolors.ENDC}")
        return
    image_np = np.asarray(Image.open(image_path))
    # corrupt returns a numpy ndarray, the same as the input
    return corrupt(image_np, corruption_name=corruption, severity=severity)

def get_safe_corruptions(corruptions = get_corruption_names()):
    return [f for f in corruptions if f not in ["glass_blur", "fog"]]

def save_image(output_path, full_path, img):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img = Image.fromarray(img)
    img.save(full_path)

def corrupt_image(img_path, img_name, 
                  severity_range=range(1,6), 
                  corruptions=get_corruption_names(), 
                  output_path="corrupted/"):
    images = {}
    uncorrupted = np.asarray(Image.open(img_path))
    # TODO: User should be able to specify where the images end up
    save_image(output_path=output_path,
               full_path=f"{output_path}/{img_name}_uncorrupted_0.png",
               img=uncorrupted)
    images['uncorrupted'] = {'0' : uncorrupted }
    for corruption in get_safe_corruptions(corruptions):
        sev_to_img = {}
        for severity in severity_range:
            log(msg=f"Corrupting {img_name} with type {corruption} and severity {severity}",
                bcolor_type=bcolors.OKBLUE)
            image = get_corrupted_image(corruption, img_path, severity)
            sev_to_img[severity] = image
            full_path = os.path.join(output_path, f"{img_name}_{corruption}_{severity}.png")

            save_image(output_path=output_path, full_path=full_path, img=image)
        images[corruption] = sev_to_img
    return images

def handle_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    rain_config = config['rain']
    for k, v in rain_config.items():
        # All lists except drop_color should be treated as ranges (start, stop, num_elements)
        if isinstance(v, list) and k != 'drop_color':
            start, stop, num = v
            rain_config[k] = list(np.linspace(start, stop, num))
        else:
            rain_config[k] = [v]
    keys = rain_config.keys()
    values = rain_config.values()

    dicts = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for d in dicts:
        d['blur_value'] = int(d['blur_value'])
    return dicts

if __name__ == "__main__":
    """
    corrupt_image(img_path="../../../sample_imgs/1.png", 
                  img_name="1.png", 
                  corruptions=get_corruption_names(),
                  output_path="own_corrupted/")
    """
    random.seed(7)
    np.random.seed(42)
    #image = cv2.imread('../../../sample_imgs/1.png')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #albumentations_corrupted()
    #get_rain_droplets('../../../sample_imgs/1.png')
    #generate_rain_droplets('../../../sample_imgs/1.png',
                           #'output/albumentations/droplets/droplets.json')
    
    
    configs = handle_config('configs/albumentations/test.yaml')
    img_last_config = func_albumentations('../../../sample_imgs/1.png',
                              config=configs[-1])
    img_last_config = Image.fromarray(img_last_config)
    img_last_config.show()
    