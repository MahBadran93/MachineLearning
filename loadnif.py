import os
import nibabel as nib 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from nibabel.testing import data_path 

dataPath = 'training/patient001/patient001_4d.nii.gz'

#nift_Img1 = os.path.join(dataPath,'patient001_4d.nii.gz')
img1 = nib.load(dataPath)

epi_img_data = img1.get_fdata()

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1,len(slices))
    
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        


#array = np.asarray(epi_img_data)
#arraych = np.reshape(array, [216,256,1])
"""
slice_0 = epi_img_data[:, :, 0,29]
slice_1 = epi_img_data[:, :, 1,29]
slice_2 = epi_img_data[:, :, 2,29]
slice_3 = epi_img_data[:, :, 3,29]
slice_4 = epi_img_data[:, :, 4,29]
slice_5 = epi_img_data[:, :, 5,29]
slice_6 = epi_img_data[:, :, 6,29]
slice_7 = epi_img_data[:, :, 7,29]
slice_8 = epi_img_data[:, :, 8,29]
slice_9 = epi_img_data[:, :, 9,29]
"""
slicePat1 = []
for t in range(30): # 29 frames, the image shape (width, heigh, # of slices , 30 frame for each slice), thats why we put 30 parameter for our array that holds 30 frame for slice 0 to animate
    slicePat1.append(epi_img_data[:, :, 0,t])

#print(slicePat1[0])
#plt.imshow(slice_0,cmap="gray", origin="lower")

fig = plt.figure() # make figure
#cmap=plt.get_cmap('jet')
im = plt.imshow(slicePat1[0], vmin=0, vmax=255,cmap="gray", origin="lower")

def updatefig(j):
    im.set_array(slicePat1[j])
    return [im]

ani = animation.FuncAnimation(fig, updatefig, frames=range(20), 
                              interval=50, blit=True)
plt.show()    
"""
for y in range(29):
    plt.imshow(slicePat1[y])
    plt.show()
"""
#show_slices([slice_0, slice_1, slice_2,slice_3,slice_4,slice_5,slice_6,slice_7,slice_8,slice_9])
#plt.suptitle("Cardiac slices for MRI image patient one")  
#plt.show()

#print(epi_img_data[:, :, 0,29])
#plt.imshow(slice_0,cmap="gray", origin="lower")
#plt.show()

