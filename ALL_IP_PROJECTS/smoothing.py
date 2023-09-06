import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import cv2 as cv
import os
import math

img=cv.imread('IMAGE_VIDEO_storage/noisy_1.jpg')
cv.imshow("NOISE-1",img)

#we use median blur, Gaussian blur to smooth image
smoothed_img=cv.medianBlur(img,3)
cv.imshow("MEDIAN BLUR",smoothed_img)

smoothed_img_1=cv.GaussianBlur(img,(3,3),0)
cv.imshow("GAUSSIAN BLUR",smoothed_img_1)

cv.waitKey(0)