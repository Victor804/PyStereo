import numpy as np
from numba import jit, njit

@jit(nopython=True, nogil=True, cache=True)
def zncc(img1, img2):
    img1_mean, img2_mean = img1.mean(), img2.mean()
    diff_img1 = (img1-img1_mean)
    diff_img2 = (img2-img2_mean)
    if np.sum(diff_img1**2) == 0 or np.sum(diff_img2**2) == 0:
        return -1
    return (np.sum(diff_img1*diff_img2))/(np.sqrt(np.sum(diff_img1**2)*np.sum(diff_img2**2)))

@jit(nopython=True, nogil=True, cache=True)
def ssd(img1, img2):
    return np.sum((img1-img2)**2)


@jit(nopython=True, nogil=True, cache=True)
def sad(img1, img2):
    return np.sum(np.absolute(img1-img2))


@jit(nopython=True, nogil=True)
def resize_image(img, x, y, dx):
    resize_img = img[(y-dx if y-dx > 0 else 0):(y+dx+1), (x-dx if x-dx > 0 else 0):(x+dx+1)]
    if resize_img.shape == (2*dx+1, 2*dx+1):
        return resize_img
    else:
        return np.eye(2*dx+1, dtype=np.uint8)


@jit(nopython=True, nogil=True)
def matching_point_zncc(img1, img2, x, y, dx, interval, direction):
    max = -1, x
    resize_img1 = resize_image(img1, x, y, dx)
    if direction == 1: end = x+interval if x+interval < len(img2[0]) else len(img2[0])
    elif direction == -1: end = x-interval if x-interval > 0 else 0
    for i in range(x, end, direction):
        r = zncc(resize_img1, resize_image(img2, i, y, dx))
        if max[0] < r:
            max = r, i
    return max[1], y


@jit(nopython=True, nogil=True)
def matching_point_ssd(img1, img2, x, y, dx, interval, direction):
    min = np.Inf, x
    resize_img1 = resize_image(img1, x, y, dx)
    if direction == 1: end = x+interval if x+interval < len(img2[0]) else len(img2[0])
    elif direction == -1: end = x-interval if x-interval > 0 else 0
    for i in range(x, end, direction):
        r = ssd(resize_img1, resize_image(img2, i, y, dx))
        if min[0] > r:
            min = r, i
    return min[1], y
