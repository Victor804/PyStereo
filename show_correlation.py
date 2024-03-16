import matplotlib.pyplot as plt
import cv2
import correlation
import numpy as np

img1 = cv2.imread("gray_1_2.png", 0)
img2 = cv2.imread("gray_2_2.png", 0)

x = 433
y = 476
attendu = 388

dx = 9

resize_img1 = correlation.resize_image(img1, x, y, dx)


score_correlation = []
for i in range(0, len(img1[0])):
    resize_img2 = correlation.resize_image(img2, i, y, dx)
    score_correlation.append(correlation.zncc(resize_img1, resize_img2))

front_title = {'family':'serif','size':20}
front = {'family':'serif','size':15}

print(max(score_correlation), score_correlation.index(max(score_correlation)))

plt.title("Mesure de corrélation SAD", fontdict = front_title)
plt.xlabel("Indice i'", fontdict = front)
plt.ylabel("Score de corrélation", fontdict = front)

plt.plot(np.array([i for i in range(len(score_correlation))][9:-9]), score_correlation[9:-9])

plt.show()
