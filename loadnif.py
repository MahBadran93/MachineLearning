import os
import nibabel as nib 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation



from nibabel.testing import data_path 

dataPath = 'training/patient001/patient001_4d.nii.gz' # path for 4d image 
dataPathgt = 'training/patient001/patient001_frame01_gt.nii.gz' # path for gt image, segmented slices, 3d


img1 = nib.load(dataPath)
img2 = nib.load(dataPathgt)

trainingImg1 = img1.get_fdata() # 4d, to show the animated cardiac image
trainingImg2 = img2.get_fdata() # to show the segmented slices 

slicePat1 = [] # array includes all the frames for animation


"""
def show_slices(slices):
    fig, axes = plt.subplots(1,len(slices))
    
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
"""        

"""
plt.imshow(trainingImg2[:,:,6]) # show the segmented slices 
plt.show()
"""
#array = np.asarray(trainingImg1)
#arraych = np.reshape(array, [216,256,1])
"""
slice_0 = trainingImg1[:, :, 0,29]
slice_1 = trainingImg1[:, :, 1,29]
slice_2 = trainingImg1[:, :, 2,29]
slice_3 = trainingImg1[:, :, 3,29]
slice_4 = trainingImg1[:, :, 4,29]
slice_5 = trainingImg1[:, :, 5,29]
slice_6 = trainingImg1[:, :, 6,29]
slice_7 = trainingImg1[:, :, 7,29]
slice_8 = trainingImg1[:, :, 8,29]
slice_9 = trainingImg1[:, :, 9,29]
"""
for t in range(30): # 29 frames, the image shape (width, heigh, # of slices , 30 frame for each slice), thats why we put 30 parameter for our array that holds 30 frame for slice 0 to animate
    slicePat1.append(trainingImg1[:, :, 0,t])

#print(slicePat1[0])

fig = plt.figure() # make figure

axes = fig.subplots(1,2) # make two subplots in the figure


#cmap=plt.get_cmap('jet')
im = axes[1].imshow(slicePat1[0], vmin=0, vmax=255,cmap="gray", origin="lower") #show 3d animated image
axes[0].imshow(trainingImg2[:,:,9])   #show segmented slice


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
