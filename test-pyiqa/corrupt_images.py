from PIL import Image
import numpy as np
import os
from utils import bcolors, log, load_yaml
import albumentations.augmentations.functional as F
import cv2
import random
import json
import itertools
import argparse

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

def generate_rain_configs(config, use_combinations, n_configs):
    rain_config = config['rain']
    for k, v in rain_config.items():
        # All lists except drop_color should be treated as ranges (start, stop, num_elements)
        if isinstance(v, list) and k != 'drop_color':
            start, stop, num = v
            rain_config[k] = list(np.linspace(start, stop, num))
        else:
            rain_config[k] = [v]
    if use_combinations:
        dicts = generate_combinations(rain_config)
    else:
        dicts = generate_n_configs(rain_config, n_configs)
    for d in dicts:
        d['randomness_seed'] = config['randomness_seed']
        d['blur_value'] = int(d['blur_value'])
    return dicts

def generate_fog_configs(config, use_combinations, n_configs):
    fog_config = config['fog']
    for k, v in fog_config.items():
        # All lists except drop_color should be treated as ranges (start, stop, num_elements)
        if isinstance(v, list):
            start, stop, num = v
            fog_config[k] = list(np.linspace(start, stop, num))
        else:
            fog_config[k] = [v]
    if use_combinations:
        dicts = generate_combinations(fog_config)
    else:
        dicts = generate_n_configs(fog_config, n_configs)
    for d in dicts:
        d['randomness_seed'] = config['randomness_seed']
    return dicts

def generate_n_configs(config, n):
    """
    Assumes that every variable in config has 1 or n possible values.
    Generates n configs.
    """
    dicts = [{} for _ in range(n)]
    for k, v in config.items():
        for i in range(n):
            if len(v) == 1:
                dicts[i][k] = v[0]
            else:
                dicts[i][k] = v[i]
    return dicts

def generate_combinations(config):
    """
    Generates all possible combinations of values specified in the config
    """
    keys = config.keys()
    values = config.values()
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]

def generate_images(img_path, output_path, configs, corrupt_func):
    log(f"Generating {len(configs)} images at {output_path}", bcolor_type=bcolors.WARNING)
    file_to_config = {}
    for i, config in enumerate(configs):
        log("Generating image with config", bcolor_type=bcolors.OKBLUE)
        log(f"{config}", bcolor_type=bcolors.OKCYAN)
        img = corrupt_func(img_path=img_path, config=config)
        img = Image.fromarray(img)
        path = os.path.join(output_path, f'{i}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
        file_to_config[f'{i}.png'] = config
    
    with open(os.path.join(output_path, 'file_to_config.json'), "w") as f:
        json.dump(file_to_config, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="pyiqa runnner")
    parser.add_argument("-c", "--config", type=str, help="path to where the config .yaml file is located")
    args = parser.parse_args()
    if not args.config:
        raise argparse.ArgumentTypeError("Directory path must be specified")
    else:
        config = load_yaml(args.config)
        img_path = config['image_path']
        use_combinations = config['use_combinations']
        n_configs = config['n_configs']
        if 'rain' in config:
            log(f"\nGenerating rainy images", bcolor_type=bcolors.HEADER)
            output_path = config['rain']['output_dir']
            rainy_configs = generate_rain_configs(config, use_combinations, n_configs)
            generate_images(img_path=img_path,
                            output_path=output_path,
                            configs=rainy_configs,
                            corrupt_func=add_rain)
        if 'fog' in config:
            log(f"\nGenerating foggy images", bcolor_type=bcolors.HEADER)
            output_path = config['fog']['output_dir']
            fog_configs = generate_fog_configs(config, use_combinations, n_configs)
            generate_images(img_path=img_path,
                            output_path=output_path,
                            configs=fog_configs,
                            corrupt_func=add_fog)

if __name__ == "__main__":
    main()