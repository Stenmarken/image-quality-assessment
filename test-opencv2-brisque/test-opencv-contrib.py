import cv2 as cv
import os
import numpy as np

path = "../../../sample_imgs"
img_paths = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
for img_path in img_paths:
    img = cv.imread(os.path.join(path, img_path), 1)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurScore = cv.Laplacian(grey, cv.CV_64F).var()
    #score = cv.quality.QualityBRISQUE_compute([img], "brisque_model_live.yml", "brisque_range_live.yml")
    obj = cv.quality.QualityBRISQUE_create("brisque_model_live.yml", "brisque_range_live.yml")
    score = obj.compute(img)
    #print(f' >> Blur Score: {blurScore}')
    print(f'Image: {img_path}, BRISQUE Score: {score}, Blur score: {blurScore}')

#cv.namedWindow("Output", cv.WINDOW_NORMAL)
#cv.imshow("Output", img)
#k = cv.waitKey(0)