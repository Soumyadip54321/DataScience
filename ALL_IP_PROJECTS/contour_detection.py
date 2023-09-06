import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import cv2 as cv
import os
import math

#read image
flower_img=cv.imread('IMAGE_VIDEO_storage/flower_1.jpg')
#cv.imshow("FLOWER",flower_img)

#define a blank image of same dimensions as the image we want to look at the contours of
blank=np.zeros(flower_img.shape,dtype='uint8')
cv.imshow("BLANK",blank)

flower_gray_img=cv.cvtColor(flower_img,cv.COLOR_BGR2GRAY)
#cv.imshow("GRAY",flower_gray_img)

#Now to compute the contours in the image we first detect the edges present either via CANNY edge detection or THRESHOLDING

#CANNY
flower_edges=cv.Canny(flower_img,125,175)
cv.imshow("CANNY EDGES",flower_edges)

contours,hierarchies=cv.findContours(flower_edges,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print("The number of contours found are {cont}".format(cont=len(contours)))

#THRESHOLDING
ret,thresh=cv.threshold(flower_gray_img,125,255,cv.THRESH_BINARY)
#cv.imshow("THRESHOLDED",thresh)
#cv.waitKey(5000)

contours,hierarchies=cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
#print("The number of contours found are {cont}".format(cont=len(contours)))
#cv.waitKey(3000)

#now we draw contours over the blank.
cv.drawContours(blank,contours,-1,(0,0,255),2)
cv.imshow("CONTOURS OF FLOWER",blank)
cv.waitKey(0)

#comapre contours and edges obtained using CANNY

