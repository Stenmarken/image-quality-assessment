import pyiqa
from PIL import Image
import numpy as np
import torch
import os

def calculate_brisque(path, file_types=()):
    """
    Calculates BRISQUE for all images with file type file_types in path.
    """
    img_paths = [f for f in os.listdir(path) if f.lower().endswith(file_types)]
    for img_path in img_paths:
        img_path = os.path.join(path, img_path)
        image = Image.open(img_path)
        # Unsqueezing is done to get 4D tensor
        img_tensor = pyiqa.utils.img_util.imread2tensor(image).unsqueeze(0)
        model = pyiqa.create_metric('brisque')

        score = model(img_tensor)

        print(f'IQA score for {img_path}: {score}')

def calculate_brisque_uniform(path, file_types=[]):
    """
    Calculates BRISQUE for all images with file type file_types in path.
    Requires uniform dimensions of images.
    """
    img_paths = [f for f in os.listdir(path) if f.lower().endswith((" ".join(file_types)))]
    images = [Image.open(os.path.join(path, img_path)) for img_path in img_paths]
    
    img_tensors = map(lambda img: pyiqa.utils.img_util.imread2tensor(img), images)
    model = pyiqa.create_metric('brisque')
    score = model(torch.stack(list(img_tensors)))
    for i in range(len(img_paths)):
        print(f'IQA score for {img_paths[i]}: {score[i]}')

#calculate_brisque_uniform("../../sample_imgs", file_types=[".png"])
calculate_brisque("../../../sample_imgs", file_types=(".jpg", ".png"))

#image = Image.open("../../../sample_imgs/")
# Unsqueezing is done to get 4D tensor
#img_tensor = pyiqa.utils.img_util.imread2tensor(image).unsqueeze(0)
#model = pyiqa.create_metric('brisque')

#score = model(img_tensor)

#print(f'IQA score for 1.png: {score}')