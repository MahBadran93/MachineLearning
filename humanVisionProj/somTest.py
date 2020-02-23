import numpy as np 
import matplotlib.pyplot as plt 
import EuclDistance as dis
import MinNeural as winMin
import UpdateWeight as updWeight
###########################################
inputData = [
    [1,1,0,0],
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,1]
]
dataInptArray = np.array(inputData) # make numpy array of out input 

############################################
"""
below to create a random data points (map), and we will start comparing the input data
(dataInptArray) with the map data(nuralsData) for classification.
"""
# neural map 
row = 2 # map data points (neurals)
col = 4 # dimension
nuralsData = np.random.rand(row,col) # the map with 2x4 

###############################################

learningRate = 0.2
iteration = 20
index = 0

for i in range(iteration):
    print('interation',i,nuralsData)
    for j in range (4):
        winner = winMin.min(nuralsData, dataInptArray[j] , index)
        updatedVector = updWeight.update(nuralsData[winner], dataInptArray[winner], learningRate)
        nuralsData[winner] = updatedVector
    learningRate = learningRate * 0.5 





