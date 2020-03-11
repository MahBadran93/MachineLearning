import numpy as np 
import pandas as pd
import csv
from io import StringIO
import matplotlib.pyplot as plt
import MinWeightData as WinnerData
import TrainData as Train


control = pd.read_fwf('control.txt')
patient = pd.read_fwf('patient.txt')


patientData = np.array(patient)
controlData = np.array(control)

trainData = np.concatenate((patientData,controlData), axis=0)
print(trainData.shape)

weightData = np.random.rand(2 , 650)

iteration = 600
learningRate = 0.6

#patientNoEmptyValues = np.array(patientData[patientData!=0])
#patieNotZeros = np.argwhere(patientData)

fig = plt.figure(figsize=(20,10))
figp = fig.subplots(2,2,gridspec_kw={'width_ratios': [8, 3]})

figp[0,0].imshow(trainData[15:16,20:30])



weightDataEnd = Train.trainData(weightData,trainData,iteration,learningRate)
print(weightData[1,25])


figp[0,1].imshow(weightDataEnd[1:2,20:30])



"""
for i in range(int(patientData.shape[1]/10.0)):
    figp[0,0].imshow(patientData[0:2,i+10:i+20])
    plt.pause(0.1)
    i = i + 10
"""

print("TESTING.................")
testData = trainData[1]

def test():
    index = 0
    for t in range(trainData.shape[0]):
        #winnerF = WinnerData.min(weightData, trainData[t],index)
        winnerTest = WinnerData.min(weightDataEnd,trainData[t],index)
        print(' class:', winnerTest)        
       



test()
#plt.show()


