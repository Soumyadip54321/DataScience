import numpy as np
import pandas as pd
import os
from skimage import io,img_as_float
import cv2 

img=img_as_float(io.imread("IMAGE_VIDEO_storage/cameraman.jpeg"))
kernel=np.ones((5,5),np.float32)/25

gaussian_kernel=np.array([[1/16,1/8,1/16],
                          [1/8,1/4,1/8],
                          [1/16,1/8,1/16]])

laplacian_kernel=np.array([[0,1,0],
                          [1,-4,-1],
                          [0,1,0]])

prewit_kernel=np.array([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]])

sobel_kernel=np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])

smoothed_img=cv2.filter2D(img,-1,gaussian_kernel,borderType=cv2.BORDER_CONSTANT)
cv2.imshow("original",img)
cv2.imshow("smoothed gaussian",smoothed_img)

smoothed_img_1=cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_CONSTANT)
cv2.imshow("smoothed box-filter",smoothed_img_1)

sharpened_img=cv2.filter2D(img,-1,laplacian_kernel,borderType=cv2.BORDER_CONSTANT)
cv2.imshow("sharpened laplacian",sharpened_img)

sharpened_img_2=cv2.filter2D(img,-1,laplacian_kernel,borderType=cv2.BORDER_CONSTANT)
cv2.imshow("sharpened with prewit",sharpened_img_2)

sharpened_img_3=cv2.filter2D(img,-1,sobel_kernel,borderType=cv2.BORDER_CONSTANT)
cv2.imshow("sharpened sobel",sharpened_img_3)

cv2.waitKey(0)
cv2.destroyAllWindows()



