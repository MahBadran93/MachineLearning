import numpy as np 
import matplotlib.pyplot as plt 
import MinWeightData as WinnerData
import TrainData as Train
 

###########################################
testingData = [
    [0,0,0,0.9],
    [0,0,0.8,0.9],
    [0.7,0,0,0],
    [0.7,0.9,0,0]
]
trainingData = [
    [1,1,0,0],
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,1]
]
trainData = np.array(trainingData) # make numpy array of out input 
testData = np.array(testingData)
weightData = np.array([[.2, .6, .5, .9],[.8, .4, .7, .3]])
#weightData = np.random.rand()
print("before trained:", weightData)


############################################
"""
below to create a random data points (map), and we will start comparing the input data
(dataInptArray) with the map data(nuralsData) for classification.
"""

###############################################

learningRate = 0.6
iteration = 1000


# give the train data function a weight matrix and a train data to process 
weightData = Train.trainData(weightData,trainData,iteration,learningRate)

print("trained:", weightData)

print("TESTING.................")
def test():
    index = 0
    for t in range(testData.shape[0]):
        #winnerF = WinnerData.min(weightData, trainData[t],index)
        winnerTest = WinnerData.min(weightData,testData[t],index,weightData.shape[0])
        print(' class:', winnerTest , 'testData:', testData[t])        
       
test()