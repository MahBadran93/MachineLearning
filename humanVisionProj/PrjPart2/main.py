import numpy as np 
import pandas as pd
import csv
from io import StringIO
import matplotlib.pyplot as plt


control = pd.read_fwf('control.txt')
patient = pd.read_fwf('patient.txt')

patientData = np.array(patient)
controlData = np.array(control)

#patientNoEmptyValues = np.array(patientData[patientData!=0])
patieNotZeros = np.argwhere(patientData)
print(patieNotZeros.shape)

fig = plt.figure(figsize=(20,10))
figp = fig.subplots(2,2,gridspec_kw={'width_ratios': [8, 3]})

figp[0,0].imshow(patientData[0:1,10:20])
figp[0,1].imshow(patientData[0:1,20:60])

figp[1,0].imshow(controlData[2:3,20:30])
figp[1,1].imshow(controlData[3:4,40:50])

# add slices check edited git 

tt = np.array([[99,  5,  2,  4],
 [ 7,  6,  8,  8],
 [ 1,  6,  7,  7]])

#print(patientData[9])

"""
for i in range(int(patientData.shape[1]/10.0)):
    figp[0,0].imshow(patientData[0:2,i+10:i+20])
    plt.pause(0.1)
    i = i + 10
"""


plt.show()

