import numpy as np 
from skimage.color import rgb2gray
import cv2 
import matplotlib.pyplot as plt 
from scipy import ndimage
from sklearn.cluster import KMeans
import mpmath as math 
from sympy.geometry import *






testImage = cv2.imread('1.jpeg')
edgeImg = cv2.imread('index.png')
ss = np.array(testImage)
cv2.circle(testImage,(int(ss.shape[0]/2),int(ss.shape[1]/2)),15,1,thickness=1)
events = [i for i in dir(cv2) if 'EVENT' in i ]
print(events)

gryImg = rgb2gray(testImage)
gryedgeImg = rgb2gray(edgeImg)

gray_r = gryImg.reshape(gryImg.shape[0]*gryImg.shape[1])

for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0            

gray = gray_r.reshape(gryImg.shape[0], gryImg.shape[1])  

###################### Edge detection 
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
 
sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
print(sobel_vertical, 'is a kernel for detecting vertical edges')

convHor = ndimage.convolve(gryedgeImg , sobel_vertical , mode="reflect")

####################### end

############################### clustering 
pic = plt.imread('1.jpeg')/255  # dividing by 255 to bring the pixel values between 0 and 1
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
kmeans = KMeans(n_clusters=8, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)



#print(clusterdPic.shape)
#plt.show()