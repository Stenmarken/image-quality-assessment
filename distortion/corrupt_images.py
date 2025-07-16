from PIL import Image
import numpy as np
from pathlib import Path
from utils.utils import bcolors, log, load_yaml

from albumentations.augmentations.pixel import functional as F
import albumentations as A
import cv2
import random
import json
import argparse


def get_image_paths(path):
    image_extensions = {".jpg", ".jpeg", ".png"}
    posix_paths = [
        f.resolve()
        for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    return list(map(str, posix_paths))


def random_k_select(path, k, seed):
    np.random.seed(seed)
    files = get_image_paths(path)
    # replace=False ensures no duplicates
    return np.random.choice(files, k, replace=False)


def generate_droplets(image, share, seed):
    np.random.seed(seed)
    height, width, _ = image.shape
    num_samples = int(share * height * width)
    indices = np.random.choice(height * width, num_samples, replace=False)

    # OpenCV wants the coordinates in (x, y) form
    coords = np.stack((indices % width, indices // width), axis=1)
    return coords


def add_rain(img_path, config):
    random.seed(config["randomness_seed"])
    image = read_img_BGR(img_path)
    droplets = generate_droplets(
        image, config["droplet_share"], config["randomness_seed"]
    )
    return F.add_rain(
        img=image,
        slant=config["slant"],
        drop_length=config["drop_length"],
        drop_width=config["drop_width"],
        drop_color=config["drop_color"],
        blur_value=config["blur_value"],
        brightness_coefficient=config["brightness_coefficient"],
        rain_drops=droplets,
    )


def add_fog(img_path, config):
    random.seed(config["randomness_seed"])
    image = read_img_BGR(img_path)
    droplets = generate_droplets(
        image, config["fog_particle_share"], config["randomness_seed"]
    )
    return F.add_fog(
        img=image,
        fog_intensity=config["fog_intensity"],
        alpha_coef=config["alpha_coef"],
        fog_particle_positions=droplets,
        fog_particle_radiuses=[config["fog_particle_size"]] * len(droplets),
    )


def add_contrast_brightness(img_path, config):
    image = read_img_BGR(img_path)
    # Necessary as the default values are non-zero
    b_limit, c_limit = (0, 0)
    if "brightness_limit" in config:
        b_limit = config["brightness_limit"]
    if "contrast_limit" in config:
        c_limit = config["contrast_limit"]
    c_b_transform = A.ReplayCompose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=b_limit,
                contrast_limit=c_limit,
                p=1,
                brightness_by_max=config["brightness_by_max"],
                ensure_safe_range=config["ensure_safe_range"],
            )
        ]
    )
    replay = c_b_transform(image=image)
    return replay["image"]


def add_blur(img_path, config):
    image = read_img_BGR(img_path)
    blur_transform = A.ReplayCompose([A.Blur(blur_limit=config["blur_value"], p=1)])
    replay = blur_transform(image=image)
    return replay["image"]


def read_img_BGR(img_path):
    return cv2.imread(img_path)


def read_img_RGB(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def generate_blur_configs(config, n_configs):
    blur_config = config["blur"]
    start, stop, step = blur_config["blur_value"]
    blurs = [(i, i) for i in range(start, stop + 1, step) if i % 2 == 1]
    return [{"blur_value": blur} for blur in blurs]


def generate_brightness_configs(config, n_configs):
    brightness_config = config["brightness"]
    start, stop, n = brightness_config["brightness_limit"]
    brightness_values = [(i, i) for i in np.linspace(start, stop, n)]
    return [
        {
            "brightness_limit": b,
            "brightness_by_max": brightness_config["brightness_by_max"],
            "ensure_safe_range": brightness_config["ensure_safe_range"],
        }
        for b in brightness_values
    ]


def generate_contrast_configs(config, n_configs):
    contrast_config = config["contrast"]
    start, stop, n = contrast_config["contrast_limit"]
    contrast_values = [(i, i) for i in np.linspace(start, stop, n)]
    return [
        {
            "contrast_limit": b,
            "brightness_by_max": contrast_config["brightness_by_max"],
            "ensure_safe_range": contrast_config["ensure_safe_range"],
        }
        for b in contrast_values
    ]


def generate_rain_configs(config, n_configs):
    rain_config = config["rain"]
    for k, v in rain_config.items():
        # All lists except drop_color should be treated as ranges (start, stop, num_elements)
        if isinstance(v, list) and k != "drop_color":
            start, stop, num = v
            rain_config[k] = list(np.linspace(start, stop, num))
        else:
            rain_config[k] = [v]
    dicts = generate_n_configs(rain_config, n_configs)
    for d in dicts:
        d["randomness_seed"] = config["randomness_seed"]
        d["blur_value"] = int(d["blur_value"])
    return dicts


def generate_fog_configs(config, n_configs):
    fog_config = config["fog"]
    for k, v in fog_config.items():
        # All lists except drop_color should be treated as ranges (start, stop, num_elements)
        if isinstance(v, list):
            start, stop, num = v
            fog_config[k] = list(np.linspace(start, stop, num))
        else:
            fog_config[k] = [v]
    dicts = generate_n_configs(fog_config, n_configs)
    for d in dicts:
        d["randomness_seed"] = config["randomness_seed"]
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


def generate_images(distortion, yaml_config, configs, corrupt_func, output_path):
    images_path = Path(yaml_config["images_path"])
    output_path = Path(output_path)
    seed = yaml_config["randomness_seed"]
    log(
        f"Generating {len(configs)} images at {output_path}",
        bcolor_type=bcolors.WARNING,
    )
    if yaml_config["select_k_random"]:
        k = yaml_config["k"]
        log(
            f"Selecting {k} images at random",
            bcolor_type=bcolors.WARNING,
        )
        images = random_k_select(images_path, k, seed)
    else:
        images = get_image_paths(images_path)

    file_to_config = {}
    for img in images:
        full_path = output_path / Path(img).name
        full_path.parent.mkdir(parents=True, exist_ok=True)
        for i, config in enumerate(configs):
            image_name = f"{distortion}_{i}"
            generate_one_image(corrupt_func, img, config, full_path, image_name)
            file_to_config[f"{Path(img).name}/{image_name}.png"] = config

    with open(output_path / "file_to_config.json", "w") as f:
        json.dump(file_to_config, f, indent=4)


def generate_one_image(corrupt_func, img_path, config, output_path, image_name):
    # log(f"Generating image with config {config}", bcolor_type=bcolors.OKBLUE)
    img = corrupt_func(img_path=img_path, config=config)
    path = output_path / f"{image_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


distortion_configs = {
    "rain": generate_rain_configs,
    "fog": generate_fog_configs,
    "blur": generate_blur_configs,
    "brightness": generate_brightness_configs,
    "contrast": generate_contrast_configs,
}
distortion_transforms = {
    "rain": add_rain,
    "fog": add_fog,
    "blur": add_blur,
    "brightness": add_contrast_brightness,
    "contrast": add_contrast_brightness,
}


def main():
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
        yaml_config = load_yaml(args.config)

        meta_keys = {
            "randomness_seed",
            "images_path",
            "n_configs",
            "select_k_random",
            "k",
        }

        for distortion in yaml_config:
            if distortion in meta_keys:
                continue
            corrupt_func = distortion_transforms[distortion]
            config_func = distortion_configs[distortion]

            log(f"\nGenerating {distortion} images", bcolor_type=bcolors.HEADER)
            output_path = yaml_config[distortion]["output_dir"]
            configs = config_func(yaml_config, yaml_config["n_configs"])
            generate_images(distortion, yaml_config, configs, corrupt_func, output_path)


if __name__ == "__main__":
    main()
