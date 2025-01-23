from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import os
from utils import bcolors, log
import albumentations.augmentations.functional as F
import cv2
import random
import json
import yaml
import itertools

def generate_droplets(image, share, seed):
    np.random.seed(seed)
    height, width, _ = image.shape
    num_samples = int(share * height*width)
    indices = np.random.choice(height*width, num_samples, replace=False)
    #TODO: Why is it (index%width, index // width). 
    # Shouldn't it be (index // width, index % width)
    coords = [(index % width, index // width) for index in indices]
    return coords

def add_rain(img_path, config):
    random.seed(config['randomness_seed'])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    droplets  = generate_droplets(image, config['droplet_share'], config['randomness_seed'])
    return F.add_rain(img=image,
               slant=config['slant'],
               drop_length=config['drop_length'],
               drop_width=config['drop_width'],
               drop_color=config['drop_color'],
               blur_value=config['blur_value'],
               brightness_coefficient=config['brightness_coefficient'],
               rain_drops=droplets)

def add_fog(img_path, config):
    random.seed(config['randomness_seed'])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    droplets  = generate_droplets(image, config['fog_particle_share'], config['randomness_seed'])
    return F.add_fog(img=image,
                      fog_intensity=config['fog_intensity'],
                      alpha_coef=config['alpha_coef'],
                      fog_particle_positions=droplets,
                      fog_particle_radiuses=[config['fog_particle_size']]*len(droplets))

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

def generate_rain_configs(file_path):
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
        d['randomness_seed'] = config['randomness_seed']
        d['blur_value'] = int(d['blur_value'])
    return dicts

def generate_fog_configs(file_path):
    config = {}
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    fog_config = config['fog']
    for k, v in fog_config.items():
        # All lists except drop_color should be treated as ranges (start, stop, num_elements)
        if isinstance(v, list):
            start, stop, num = v
            fog_config[k] = list(np.linspace(start, stop, num))
        else:
            fog_config[k] = [v]
    keys = fog_config.keys()
    values = fog_config.values()

    dicts = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for d in dicts:
        d['randomness_seed'] = config['randomness_seed']
    return dicts

def generate_images(config_path, img_path, output_path):
    configs = generate_rain_configs(config_path)
    #configs = generate_fog_configs(config_path)
    log(f"Generating {len(configs)} images", bcolor_type=bcolors.WARNING)
    file_to_config = {}
    for i, config in enumerate(configs):
        log("Generating image with config", bcolor_type=bcolors.OKBLUE)
        log(f"{config}", bcolor_type=bcolors.OKCYAN)
        img = add_rain(img_path=img_path, config=config)
        #img = add_fog(img_path=img_path, config=config)
        img = Image.fromarray(img)
        path = os.path.join(output_path, f'{i}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
        file_to_config[f'{i}.png'] = config
    
    with open(os.path.join(output_path, 'file_to_config.json'), "w") as f:
        json.dump(file_to_config, f, indent=4)

if __name__ == "__main__":
    generate_images(config_path='configs/albumentations/test.yaml',
                    img_path='../../../sample_imgs/1.png',
                    output_path='output/albumentations/rainy_images')
    