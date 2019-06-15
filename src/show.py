import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
# import math
# import cv2
# # fig, (ax1) = plt.subplots(1,1)
img = np.loadtxt("output_src.txt").reshape(3,400,400)
img = img.transpose(2,1,0)
plt.imshow(img.astype("uint8"))
# img = Image.fromarray(img.astype("uint8"), mode="HSV")
# img=img.convert("RGB")
# img.show()
plt.show()

# # img = np.array(img)
# # print(img)
# # img[:,:,0] /= 255;
# # img[:,:,0] *= 360;
# # img = Image.fromarray(img.astype("int"), "HSV")
# img.show()
# # ax1.imshow(img.astype("uint8"))
# # img = np.loadtxt("rgb.txt").reshape(3,215,289)
# # img = img.transpose(2,1,0)
# img = 
# # # clahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(16,16))
# # # cv2.imwrite("head.jpg", img)
# # # img = cv2.imread("head.jpg")
# # # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # # img = np.concatenate([img[:,:,:2],np.expand_dims(clahe.apply(img[:,:,2]), 2)],2)
# # ax2.imshow(np.array(img).astype("uint8"))

# fig, axes = plt.subplots(2,2)
# img1 = plt.imread("./head.png")
# img2 = plt.imread("./head-sharpen.png")
# img3 = plt.imread("./head-blur.png")
# img4 = plt.imread("./head-eh.png")
# axes[0,0].imshow(img1)
# axes[0,0].set_title("原始图片")
# axes[0,0].axis("off")
# axes[0,1].imshow(img2)
# axes[0,1].set_title("锐化图片")
# axes[0,1].axis("off")
# axes[1,0].imshow(img3)
# axes[1,0].set_title("模糊图片")
# axes[1,0].axis("off")
# axes[1,1].imshow(img4)
# axes[1,1].set_title("增强图片")
# axes[1,1].axis("off")
# plt.show()