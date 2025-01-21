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
        p=1.0
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
    droplets = get_droplets(config['droplets_path'])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return F.add_rain(img=image,
               slant=config['slant'],
               drop_length=config['drop_length'],
               drop_width=config['drop_width'],
               drop_color=config['drop_color'],
               blur_value=config['blur_value'],
               brightness_coefficient=config['brightness_coefficient'],
               rain_drops=droplets)

def albumentations_corrupted():
    drop_lengths = [20, 40, 60]
    blur_values = [5, 7, 9]
    drop_widths = [1, 2, 3]
    brightness_coefficients = [0.7, 0.5, 0.3]

    image = cv2.imread('../../../sample_imgs/1.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(3):
        transform = A.ReplayCompose([A.RandomRain(
                    slant_range=(-15, -15),
                    drop_length=drop_lengths[i],
                    drop_width=drop_widths[i],
                    drop_color=(180, 180, 180),
                    blur_value=blur_values[i],
                    brightness_coefficient=brightness_coefficients[i],
                    p=1.0,
                    rain_type='heavy')])
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

if __name__ == "__main__":
    """
    corrupt_image(img_path="../../../sample_imgs/1.png", 
                  img_name="1.png", 
                  corruptions=get_corruption_names(),
                  output_path="own_corrupted/")
    """
    random.seed(7)
    np.random.seed(42)
    albumentations_corrupted()
    #get_rain_droplets('../../../sample_imgs/1.png')
    #generate_rain_droplets('../../../sample_imgs/1.png',
                           #'output/albumentations/droplets/droplets.json')
    """
    config = {'droplets_path': 'output/albumentations/droplets/droplets.json',
              'slant': 10,
              'drop_length' : 8,
              'drop_width' : 1,
              'drop_color' : (74,101,131),
              'blur_value' : 1,
              'brightness_coefficient' : 0.7}
    img = func_albumentations('../../../sample_imgs/1.png',
                              config=config)
    img = Image.fromarray(img)
    img.show()
    """