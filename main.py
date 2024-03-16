import cv2
import correlation
import numpy as np
from numba import njit, prange

def map(x, from_min, from_max, to_min, to_max):
    return (x-from_min)*(to_max-to_min)/(from_max-from_min)+to_min


@njit(parallel=True)
def disparity(image1, image2, interval, dx, matching_point):
    disp = np.zeros((len(image1), len(image1[0])))
    for j in prange(len(image1)):
        for i in range(len(image1[0])):
            x, y = matching_point(image1, image2, i, j, dx, interval, -1)
            if matching_point(image2, image1, x, y, dx, interval, 1) == (i, j):
                disp[j, i] = (i-x)
    return disp


@njit(parallel=True)
def occulation(image1, image2, interval, dx, matching_point):
    occult = np.zeros((len(image1), len(image1[0])))
    for j in prange(len(image1)):
        for i in range(len(image1[0])):
            x, y = matching_point(image1, image2, i, j, dx, interval, -1)
            if matching_point(image2, image1, x, y, dx, interval, 1) != (i, j):
                occult[j, i] = 255
    return occult

if __name__ == "__main__":
    coeff = 2.5

    interval = 30*coeff
    dx = 14

    image_name = "art"
    methode_name = "zncc"

    illum = "1"

    if methode_name == "zncc":methode = correlation.matching_point_zncc
    elif methode_name == "ssd":methode = correlation.matching_point_ssd
    """
    image1 = cv2.imread("./images/{}/illum{}/view1.png".format(image_name, illum), 0)

    for i in range(2, 3):
        image2 = cv2.imread("./images/{}/illum{}/view{}.png".format(image_name, illum, i), 0)
        disp = disparity(image1, image2, interval, dx, methode)
        image_depth = map(disp, disp.min(), disp.max(), 10, 255)
        #cv2.imwrite("./result/{}/illum{}/{}/{}.jpg".format(image_name,illum, methode_name, i), image_depth)
        cv2.imwrite("1.jpg", image_depth)
    """
    image1 = cv2.resize(cv2.imread("images/art/illum1/view0.png", 0), (round(278*coeff), round(222*coeff)))
    image2 = cv2.resize(cv2.imread("images/art/illum1/view2.png", 0), (round(278*coeff), round(222*coeff)))
    disp = occulation(image1, image2, interval, dx, methode)
    #image_depth = map(disp, disp.min(), disp.max(), 0, 255)
    cv2.imwrite("1.png", disp)
