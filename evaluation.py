import cv2
import numpy as np


def ecart_type(img1, img2):
    diff = img2-img1
    mean = diff.mean()
    variance = np.sum((diff-mean)**2)/((len(diff)*len(diff[0]))**2)
    return variance


image1 = cv2.imread("result/books/illum1/zncc/1.jpg", 0)
image2 = cv2.imread("result/books/illum1/zncc/6.jpg", 0)
image_ref = cv2.imread("images/books/disp1.png", 0)

print(ecart_type(image1, image_ref))
print(ecart_type(image2, image_ref))
"""
for i in range(0, 3):
    for y in range(1, 4):
       image2 = cv2.imread("result/art/illum{}/image_depth_art_0_{}.jpg".format(i, y), 0)

       print("illum{}, image{}".format(i, y))
       print(ecart_type(image1, image2))
"""
