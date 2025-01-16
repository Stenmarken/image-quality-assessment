from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import os
from utils import bcolors
import pyiqa.utils

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

def save_image(img_path, img):
    img = Image.fromarray(img)
    img.save(img_path)

def corrupt_image(img_path, img_name, severity_range=range(1,6), corruptions=get_corruption_names()):
    images = {}
    for corruption in get_safe_corruptions(corruptions):
        sev_to_img = {}
        for severity in severity_range:
            image = get_corrupted_image(corruption, img_path, severity)
            sev_to_img[severity] = image
            save_image(f"corrupted/{img_name}_{corruption}_{severity}.png", image)
        images[corruption] = sev_to_img
    return images