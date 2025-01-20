from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import os
from utils import bcolors, log

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
    corrupt_image(img_path="../../../sample_imgs/1.png", 
                  img_name="1.png", 
                  corruptions=get_corruption_names(),
                  output_path="own_corrupted/")