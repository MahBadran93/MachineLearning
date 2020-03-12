import numpy as np 
import pandas as pd
import csv
from io import StringIO
import matplotlib.pyplot as plt

import Funcs as F

#----------------------Read Data Set -------------------------------
control = pd.read_fwf('control.txt', header = None, delimiter="\t")
patient = pd.read_fwf('patient.txt', header = None, delimiter="\t")
test = np.loadtxt('Test/test_five.txt')
#test = pd.read_fwf('Test/test_five.txt', header = None, delimiter="\t")
#----------------------Shape Data set as arrays --------------------
patientData = np.array(patient)
controlData = np.array(control)
testData =  np.array(test)
print(testData.shape)
#----------------------Concatenate data vertically------------------
trainData = np.vstack((patientData,controlData))
#print(trainData.shape)
#--------------------Generate Random weight data -------------------
weightData = np.random.rand(2 , 650)
#---------- set initial Parameters ---------------------------------
iteration = 500
learningRate = 0.6
#-------------------------------------------------------------------

#-------------Plot Results -----------------------------------------
fig = plt.figure(figsize=(20,10))
figp = fig.subplots(2,2,gridspec_kw={'width_ratios': [8, 3]})

figp[0,0].imshow(trainData[15:16,20:30])
figp[0,1].imshow(weightData[1:2,20:30])
#plt.show()
#--------------------------Training---------------------------------
weightDataEnd = F.trainData(weightData,trainData,iteration,learningRate)



#print(testData.shape)

#........................TESTING..............................................
F.Test(test , weightDataEnd)




