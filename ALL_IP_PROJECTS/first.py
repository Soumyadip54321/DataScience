import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import cv2 as cv
import os
import math

#now we read picture here
park_img = cv.imread('IMAGE_VIDEO_storage/park_2.jpg')
cv.imshow('PARK IN BOSTON',park_img)


#converting to grayscale
gray=cv.cvtColor(park_img,cv.COLOR_BGR2GRAY)
cv.imshow('PARK GRAY',gray)


#image BLURRING
#HERE WE USE GAUSSIAN BLUR WITH A KERNEL SIZE OF (3,3)
blur = cv.GaussianBlur(park_img,(7,7),cv.BORDER_DEFAULT)
cv.imshow("PARK BLURRED",blur)


#We use Canny edge detection technique to figure what the edges are in an image.
edged_img=cv.Canny(park_img,125,175)
cv.imshow("EDGED IMAGE",edged_img)

#if applied on a blurred image, the no of edges detected reduces drastically
blur_edged_img=cv.Canny(blur,125,175)
cv.imshow("EDGES IN BLURRED IMAGE",blur_edged_img)

#we can also resize images with cv
resized_img=cv.resize(park_img,(100,100),interpolation=cv.INTER_AREA)
cv.imshow("RESIZED",resized_img)

#display dimension of image
width=park_img.shape[1]
height=park_img.shape[0]
print("THE DIMENSIONS ARE: ",width,height)

#crop image
park_cropped=park_img[300:700,80:400]
cv.imshow("CROPPED",park_cropped)

#here we transplate image
#x-->right,y-->down,-x-->left,-y-->up

#we create a function here
def translate(img,x,y):
    translation_mtx=np.float32([[1,0,x],[0,1,y]])
    dimensions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,translation_mtx,dimensions)

translated_img=translate(park_img,100,100)
cv.imshow("TRANSLATED",translated_img)

cv.waitKey(8000)

#here we rotate an image by theta angle about a rotation pt.
def rotate(img,angle,rotpoint=None):
    (width,height)=img.shape[:2]

    if rotpoint==None:
        rotpoint=(width//2,height//2)
    
    rotMat=cv.getRotationMatrix2D(rotpoint,angle,1.0)

    dimensions=(width,height)
    return cv.warpAffine(img,rotMat,dimensions)

rotated_img=rotate(park_img,-45)
cv.imshow("ROTATED CLOCKWISE",rotated_img)
cv.waitKey(8000)

#flipping image
flipped_img=cv.flip(park_img,-1)
cv.imshow("HORIZONTAL-VERTICAL-FLIP",flipped_img)
cv.waitKey(8000)



