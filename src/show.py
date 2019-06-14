import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
fig, (ax1,ax2) = plt.subplots(1,2)
img = np.loadtxt("out.txt").reshape(3,2560,1600)
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
img = np.loadtxt("rgb.txt").reshape(3,2560,1600)
img = img.transpose(2,1,0)
ax2.imshow(img.astype("uint8"))
plt.show()