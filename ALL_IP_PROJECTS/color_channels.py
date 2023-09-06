import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import cv2 as cv
import os
import math

img=cv.imread('IMAGE_VIDEO_storage/park_2.jpg')
cv.imshow("PARK",img)

#now we split the image into 3 seperate channels i.e. R,G & B.
b,g,r=cv.split(img)

#cv.imshow("BLUE",b)
#cv.imshow("GREEN",g)
#cv.imshow("RED",r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

#merge the above 3 channels
merge_img=cv.merge([b,g,r])
cv.imshow('MERGED',merge_img)

#to visualize the respective channels we create a blank img with equal dimensions and set it equal to the channels other than the color for each channels
blank=np.zeros(img.shape[:2],dtype='uint8')

blue=cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])

cv.imshow("BLUE_CHANNEL",blue)
cv.imshow("GREEN_CHANNEL",green)
cv.imshow("RED_CHANNEL",red)

cv.waitKey(0)