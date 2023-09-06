from matplotlib import pyplot as plt
import cv2 as cv

#----------------------READING A COLOUR IMAGE TO CONVERT INTO GRAYSCALE---------------------------------------------------------
park_img_rgb=plt.imread('IMAGE_VIDEO_storage/kittens.jpg')

#------------------------------SPLIT INTO RED,GREEN AND BLUE CHANNELS RESPECTIVELY AND DISPLAY IMAGE-----------------------------------------------
r,g,b=park_img_rgb[:,:,0],park_img_rgb[:,:,1],park_img_rgb[:,:,2]

park_gray=((r/3)+(b/3)+(g/3))
fig=plt.figure(1)
img1,img2=fig.add_subplot(121),fig.add_subplot(122)

img1.imshow(park_img_rgb)
img2.imshow(park_gray,cmap=plt.cm.get_cmap('gray'))
fig.show()
plt.show()

#--------------------------------------COUNT  PIXEL FREQUENCIES TO DISPALY HISTOGRAM--------------------------------------------
pixel_count={}
for i in range(256):
    pixel_count[i]=0

for i in range(park_gray.shape[0]):
    for j in range(park_gray.shape[1]):
        pixel_count[int(park_gray[i][j])]+=1

#-----------------------------------PLOT HISTOGRAM-----------------------------------------------------------------------------------
plt.bar(pixel_count.keys(),pixel_count.values(),color="red")
plt.title("Histogram of park gray image")
plt.xlabel("grayscales")
plt.ylabel("pixels per grayscale")
plt.show()
plt.close()
total_pixels=park_gray.shape[0]*park_gray.shape[1]

#-----------------------------------------FUNCTION TO COMPUTE PROBABILITY AND MEAN OF BACKGROUND AND FOREGROUND PIXELS--------------------------

def compute_probability_mean(grayscale,pixel_count):
    sum_pixels=0
    weighted_sum=0
    for gray_val in range(grayscale+1):                 #computes background & foreground probability & mean
        sum_pixels+=pixel_count[gray_val]
        weighted_sum+=gray_val*pixel_count[gray_val]
    
    background_pixel_prob=sum_pixels/(park_gray.shape[0]*park_gray.shape[1])
    try:
        bckgrnd_mean=weighted_sum/sum_pixels
    except ZeroDivisionError:
        bckgrnd_mean=0

    sum_pixels=0
    weighted_sum=0
    for gray_val in range(grayscale+1,256):
        sum_pixels+=pixel_count[gray_val]
        weighted_sum+=gray_val*pixel_count[gray_val]
    
    foreground_pixel_prob=sum_pixels/(park_gray.shape[0]*park_gray.shape[1])

    try:
        frgrnd_mean=weighted_sum/sum_pixels
    except ZeroDivisionError:
        frgrnd_mean=0
    return background_pixel_prob,foreground_pixel_prob,bckgrnd_mean,frgrnd_mean

#-------------------------------------------------------OTSU method on histogram------------------------------------------------
start_grayscale=0
inter_class_var=[]          #contains all inter_class variances computed

for i in pixel_count:
    Wb,Wf,mu_b,mu_f=compute_probability_mean(i,pixel_count)
    inter_class_var.append(Wb*Wf*(mu_b-mu_f)**2)

""" print(inter_class_var) """
#threshold is the maximum inter_class variance value which maximizes classes

#---------------------------------------------COMPUTES IDEAL THRESHOLD TO SEGMENT IMAGE---------------------------------------
max_inter_var=-1
for var_val in inter_class_var:
    if var_val>max_inter_var:
        max_inter_var=var_val

img_threshold=inter_class_var.index(max_inter_var)
print("The maximum inter-class variance and threshold is: ",max_inter_var,img_threshold)
""" print(pixel_count) """

#now we segent image into 2 classes i.e. background & foreground with all pixels intensities below threshold obtained made 0 and the rest 255

#----------------------------------------------CREATING SEGMENTED IMAGE------------------------------------------------------------
for i in range(park_gray.shape[0]):
    for j in range(park_gray.shape[1]):
        if park_gray[i][j]<=img_threshold:
            park_gray[i][j]=0
        else:
            park_gray[i][j]=255

print(park_gray,park_gray.shape)
cv.imshow("binarized image",park_gray)
cv.waitKey(0)



