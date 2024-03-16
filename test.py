import cv2
import numpy as np


image1 = cv2.imread("/home/victor/Documents/projets/stereoscopie/images/books/illum1/view0.png", 0)#Gauche
image2 = cv2.imread("/home/victor/Documents/projets/stereoscopie/images/books/illum0/view1.png", 0)#Gauche


cv2.imwrite("gray_1.png", image1)
cv2.imwrite("gray_2.png", image2)
