from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import torch as th
import numpy as np
n=9
if n==4:
    img_1_1 = th.tensor(np.transpose(plt.imread("1.jpg"),(2,1,0)))
    img_1_2 = th.tensor(np.transpose(plt.imread("2.jpg"),(2,1,0)))
    img_2_1 = th.tensor(np.transpose(plt.imread("3.jpg"),(2,1,0)))
    img_2_2 = th.tensor(np.transpose(plt.imread("4.jpg"),(2,1,0)))

    images = make_grid([img_1_1, img_1_2,img_2_1, img_2_2], nrow=2)
elif n==9:
    img_1_1 = th.tensor(np.transpose(plt.imread("1.jpg"),(2,1,0)))
    img_1_2 = th.tensor(np.transpose(plt.imread("2.jpg"),(2,1,0)))
    img_1_3 = th.tensor(np.transpose(plt.imread("3.jpg"),(2,1,0)))
    img_2_1 = th.tensor(np.transpose(plt.imread("4.jpg"),(2,1,0)))
    img_2_2 = th.tensor(np.transpose(plt.imread("5.jpg"),(2,1,0)))
    img_2_3 = th.tensor(np.transpose(plt.imread("6.jpg"),(2,1,0)))
    img_3_1 = th.tensor(np.transpose(plt.imread("7.jpg"),(2,1,0)))
    img_3_2 = th.tensor(np.transpose(plt.imread("8.jpg"),(2,1,0)))
    img_3_3 = th.tensor(np.transpose(plt.imread("9.jpg"),(2,1,0)))

    images = make_grid([img_1_1, img_1_2,img_1_3,img_2_1, img_2_2,img_2_3,img_3_1, img_3_2,img_3_3], nrow=3)

images = np.transpose(images,(2,1,0))
figpath = 'generated_CVAE.png'
fig = plt.figure('hi')
plt.axis("off")
plt.imshow(images, interpolation='nearest')
plt.savefig(figpath)
