import cv2
import numpy as np
from numba import njit, jit

@njit
def in_map(x, y, img):
    return 0<=x<len(img[0]) and  0<=y<len(img)


def fill_image(img):
    new_image = np.copy(img)
    for i in range(len(img[0])):
        for j in range(len(img)):
            if img[j][i] == 0:
                points = []
                for di, dj in zip([1, 1, 0, -1, -1, -1, 0, 1], [0, -1, -1, -1, 0, 1, 1, 1]):
                    if in_map(i+di, j+dj, img):
                        if img[j+dj][i+di] > 0:
                            points.append(img[j+dj][i+di])

                if len(points)>=3:
                    new_image[j][i] = sum(points)/len(points)
    return new_image




img = cv2.imread("1.png", 0)
new_img = fill_image(img)
i = 0
while img.any() == new_img.any():
    img = np.copy(new_img)
    new_img = fill_image(img)
    cv2.imwrite("new_img{}.jpg".format(i), new_img)
    i+=1
