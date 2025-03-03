from PIL import Image
import numpy as np
import os
from utils import bcolors, log, load_yaml
import albumentations.augmentations.functional as F
import albumentations as A
import cv2
import random
import json
import itertools
import argparse
from pathlib import Path

def random_n_select(path, n, seed):
    np.random.seed(seed)
    files = [f.resolve() for f in path.iterdir() if f.is_file()]
    # replace=False ensures no duplicates
    return np.random.choice(files, n, replace=False)

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
    image = read_img(img_path)
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
    image = read_img(img_path)
    droplets  = generate_droplets(image, config['fog_particle_share'], config['randomness_seed'])
    return F.add_fog(img=image,
                      fog_intensity=config['fog_intensity'],
                      alpha_coef=config['alpha_coef'],
                      fog_particle_positions=droplets,
                      fog_particle_radiuses=[config['fog_particle_size']]*len(droplets))

def add_blur(img_path, config):
    image = read_img(img_path)
    blur_transform = A.ReplayCompose([A.Blur(blur_limit=config['blur_value'], p=1)])
    replay = blur_transform(image=image)
    assert_applied_blur(replay)
    return replay['image']

def assert_applied_blur(replay):
    transforms = replay['replay']['transforms'][0]
    blur_value = transforms['blur_limit']
    kernel_size = transforms['params']['kernel']
    assert blur_value[0] == kernel_size
    assert transforms['applied'] == True

def read_img(img_path):
    image = cv2.imread(img_path)
    # TODO: Is it correct to do BGR2RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def generate_blur_configs(config, use_combintions, n_configs):
    blur_config = config['blur']
    start, stop, step = blur_config['blur_value']
    blurs = [(i, i) for i in range(start, stop+1, step) if i % 2 == 1]
    return [{'blur_value' : blur} for blur in blurs]

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

def generate_images(yaml_config, configs, corrupt_func, output_path):
    if 'select_random_imgs_from_base_path' in yaml_config:
        corrupt_images_batch(yaml_config, configs, corrupt_func, output_path)
    else:
        corrupt_single_image(yaml_config, configs, corrupt_func, output_path)

def corrupt_images_batch(yaml_config, configs, corrupt_func, output_path):
    base_path = Path(yaml_config['base_path'])
    output_path = Path(output_path)
    seed = yaml_config['randomness_seed']
    n = yaml_config['num_images']
    log(f"Generating {len(configs)} images at {output_path}", bcolor_type=bcolors.WARNING)
    images = random_n_select(base_path, n, seed)
    file_to_config = {}
    for img in images:
        img_path = Path(img)
        full_path = output_path / img_path.name
        full_path.parent.mkdir(parents=True, exist_ok=True)
        for i, config in enumerate(configs):
            suffix = f"blur_{config['blur_value'][0]}_kernel"
            generate_one_image(corrupt_func, i, img_path, config, full_path, suffix=suffix)
            file_to_config[f'{img_path.name}_{suffix}{i}.png'] = config
    
    with open(os.path.join(output_path, 'file_to_config.json'), "w") as f:
        json.dump(file_to_config, f, indent=4)

def corrupt_single_image(yaml_config, configs, corrupt_func, output_path):
    img_path = yaml_config['image_path']
    log(f"Generating {len(configs)} images at {output_path}", bcolor_type=bcolors.WARNING)
    file_to_config = {}
    for i, config in enumerate(configs):
        generate_one_image(corrupt_func, i, img_path, config, output_path)
        file_to_config[f'{i}.png'] = config
    
    with open(os.path.join(output_path, 'file_to_config.json'), "w") as f:
        json.dump(file_to_config, f, indent=4)

def generate_one_image(corrupt_func, i, img_path, config, output_path,
                       suffix=""):
    log(f"Generating image with config {config}", bcolor_type=bcolors.OKBLUE)
    img = corrupt_func(img_path=img_path, config=config)
    img = Image.fromarray(img)
    path = os.path.join(output_path, f'{i}{suffix}.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

distortion_configs = {'rain' : generate_rain_configs,
                      'fog' : generate_fog_configs,
                      'blur' : generate_blur_configs}
distortion_transforms = {'rain': add_rain,
                         'fog': add_fog,
                         'blur': add_blur}

def main():
    parser = argparse.ArgumentParser(description="pyiqa runnner")
    parser.add_argument("-c", "--config", type=str, help="path to where the config .yaml file is located")
    args = parser.parse_args()
    if not args.config:
        raise argparse.ArgumentTypeError("Directory path must be specified")
    else:
        yaml_config = load_yaml(args.config)
        use_combinations = yaml_config['use_combinations']
        n_configs = yaml_config['n_configs']

        for distortion in ['rain', 'fog', 'blur']:
            if distortion in yaml_config:
                log(f"\nGenerating {distortion} images", bcolor_type=bcolors.HEADER)
                output_path = yaml_config[distortion]['output_dir']
                config_func = distortion_configs[distortion]
                configs = config_func(yaml_config, use_combinations, n_configs)
                corrupt_func = distortion_transforms[distortion]
                generate_images(yaml_config, configs, corrupt_func, output_path)

if __name__ == "__main__":
    main()