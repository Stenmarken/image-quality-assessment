from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from test_pyiqa import bcolors

def get_corrupted_image(corruption, image_path, severity):
    if not os.path.isfile(image_path):
        raise f"{image_path} is not a valid file. Crashing"
    if corruption in ["glass_blur", "fog"]:
        print(f"{bcolors.WARNING}Corruption type {corruption} is not supported. Returning{bcolors.ENDC}")
        return

    image = np.asarray(Image.open(image_path))
    # corrupt returns a numpy ndarray, the same as the input
    return corrupt(image, corruption_name=corruption, severity=severity)

def get_safe_corruptions(corruptions):
    [f for f in corruptions if f not in ["glass_blur", "fog"]]

def corrupt_image(image_path, severity, corruptions=get_corruption_names()):
    image_path = "../../../sample_imgs/1.png"
    images = {}
    for corruption in get_safe_corruptions(corruptions):
        image = get_corrupted_image(corruption, image_path, severity)
        images[corruption] = image
    return images