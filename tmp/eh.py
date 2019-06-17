from cv2 import equalizeHist
import cv2
import numpy as np
import cProfile

def test():
    img = cv2.imread("./nebulous.jpg")
    img = np.concatenate([img[:,:,:2], np.expand_dims(equalizeHist(img[:,:,2]),2)],2)
    cv2.imshow("eh",img)

cProfile.run("test()")