import cv2

#img = cv2.imread("../../../sample_imgs/1.png")
img = cv2.imread("test_32_32.jpg")
blurred = []
blur_image = img
for i in range(1, 100, 2):
    blur_image = cv2.GaussianBlur(blur_image, (i, i), 0)
    blurred.append(blur_image)
#cv2.imshow('Original Image', img)
#cv2.imshow('Blur Image', blur_image)
for idx in range(20):
    cv2.imshow(f'Blur Image {idx}', blurred[idx])
#cv2.imshow(f'Blur Image {len(blurred) - 1}', blurred[-1])
#cv2.imshow(f'Blur Image {0}', blurred[0])

cv2.waitKey(0)