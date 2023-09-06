import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import cv2 as cv
import os
import math

flw2_img=cv.imread('IMAGE_VIDEO_storage/flower_2.jpg')
cv.imshow("FLOWER",flw2_img)

#OPEN CV SHOWS ANY RGB IMAGE AS BGR WHEREAS any other library such as matplotlib considers color image in regular RGB format.
#So we gotta be mindful of these subtle differences when using OPEN_CV over other image processing libraries.
#plt.imshow(flw2_img)
#plt.show()

#convert BGR to RGB images
flower_regular_img=cv.cvtColor(flw2_img,cv.COLOR_BGR2RGB)
cv.imshow("RGB",flower_regular_img)

#convert BGR to HSV
hsv_img=cv.cvtColor(flw2_img,cv.COLOR_BGR2HSV)
cv.imshow("HSV",hsv_img)


cv.waitKey(0)
