import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
import cv2
fig, (ax1) = plt.subplots(1,1)
img = np.loadtxt("out.txt").reshape(3,400,400)
img = img.transpose(2,1,0)
# img = Image.fromarray(img.astype("uint8"))
# img=img.convert("HSV")
# img.show()
# img = np.array(img)
# print(img)
# img[:,:,0] /= 255;
# img[:,:,0] *= 360;
# img = Image.fromarray(img.astype("int"), "HSV")
# img.show()
ax1.imshow(img.astype("uint8"))
# img = np.loadtxt("rgb.txt").reshape(3,215,289)
# img = img.transpose(2,1,0)
# # clahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(16,16))
# # cv2.imwrite("head.jpg", img)
# # img = cv2.imread("head.jpg")
# # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # img = np.concatenate([img[:,:,:2],np.expand_dims(clahe.apply(img[:,:,2]), 2)],2)
# ax2.imshow(np.array(img).astype("uint8"))
plt.show()