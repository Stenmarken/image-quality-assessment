import random
import os
import math
import subprocess


def create_sample(bucket_name, seed, base_dir, sub_dirs, share, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    random.seed(seed)
    for dir in sub_dirs:
        full_dir = os.path.join(base_dir, dir)
        files = sorted([f for f in os.listdir(full_dir) if os.path.isfile(os.path.join(full_dir, f))])
        num_imgs = math.ceil(len(files) * share)
        s = random.sample(files, num_imgs)
        for img in s:
            img_path = os.path.join(full_dir, img)
            img_output_path = f"{output_dir}/{dir}_{img}"
            subprocess.run(['cp', img_path, img_output_path])
    with open(f"{output_dir}/README.txt", 'w') as f:
        f.write(f"Bucket name: {bucket_name}\n")
        f.write(f"Randomness seed: {seed}\n")
        f.write(f"Image directory: {base_dir}\n")
        f.write(f"Subdirectories: {sub_dirs}\n")
        f.write(f"Share of images sampled: {share}")
    
if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    base_dir = '../../../data/tmp/avl-snow-day/dgt_2025-01-08-14-37-21_0_s0/'
    sub_dirs = ['RTB_Camera_Front', 'RTB_Camera_Rear', 'RTB_Camera_Left', 'RTB_Camera_Right']
    # The share of images to be sampled in each folder
    share = 0.02
    output_dir = '../../../data/sample_avl_snow_day/'
    bucket_name = 'roadview-avl-snow-day'

    create_sample(bucket_name, seed, base_dir, sub_dirs, share, output_dir)
