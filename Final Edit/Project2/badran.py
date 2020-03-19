#.................Result Subjects ID's........................................
"""
Test data subject 1 belongs to :  control
Test data subject 2 belongs to :  patient
Test data subject 3 belongs to :  control
Test data subject 4 belongs to :  patient

"""
#.................Student Name...............................
"""
my name : Mahmoud Badran
Project 2 

"""
#...................... Imported libraries and files...................
import numpy as np 
import pandas as pd
import csv
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid

import Funcs as F # import Fncs.py file which includes all the main functios needed


#----------------------Read Data Set -------------------------------
control = pd.read_fwf('TrainingData/control.txt', header = None, delimiter="\t")
patient = pd.read_fwf('TrainingData/patient.txt', header = None, delimiter="\t")
test = np.loadtxt('TestingData/badran.txt')

#----------------------Shape Data set as arrays --------------------
patientData = np.array(patient)
controlData = np.array(control)
testData =  np.array(test)
print('shape of the test data: ' , testData.shape)

#----------------------Concatenate data vertically------------------
trainData = np.vstack((patientData,controlData)) #patient : 0 , control : 1

#--------------------Generate weight data -------------------

#weightData = np.random.rand(2 , 650)
#np.savetxt('weightData.txt',weightData , fmt='%s')
"""
wanted a fixed weight data, so at first I created random weightdata with shape (2 , 650),
the I saved it into a weightData.txt file and saved it in the project folder, beccause I don't want the weightdata to be changing
every run
"""
weightDataLoad = np.loadtxt('weightData.txt')
weightData = np.array(weightDataLoad)
print('weight data shape  :' , weightData.shape)

#---------- set initial Parameters ---------------------------------
iteration = 30
learningRate = 0.6
#-------------------------------------------------------------------

#--------------------------Training---------------------------------
weightDataEnd = F.trainData(weightData,trainData,iteration,learningRate)
#np.savetxt('weightDataend.txt',weightDataEnd , fmt='%s')

#-------------Plot Results -----------------------------------------
fig = plt.figure(constrained_layout=True,figsize=(20,10))
gs = grid.GridSpec(2, 2,width_ratios=[2,2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])
ax1.set_title('training data(patient from 0-10 rows, control from 10-20 rows)')
ax1.imshow(trainData[:,100:150])
ax2.set_title('testing data: ')
ax2.imshow(testData[:,100:150])
ax3.set_title('weightData before training: ')
ax3.imshow(weightData[:,100:150])
ax4.set_title('weightData Trained: ')
ax4.imshow(weightDataEnd[:,100:150])
#plt.show() 
# here I was just trying to visualize the data and test 
#..............................................................................

#........................TESTING..............................................
F.Test(testData , weightDataEnd)






