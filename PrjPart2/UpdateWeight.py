import numpy as np 


def update(weightData, trainingData, learningRate):
    # weightData : parameter to pass the weightData winner vector for weighData array, weightData[winner].
    # trainingData : parameter to pass the trainingData[index], index of trainingData we are testing currently.
    # learningRate : pass the learingRate.
    
    updatedValues = np.zeros((650))
    for i in range(len(trainingData)):
        updatedValues[i] =  weightData[i] + learningRate*(trainingData[i] - weightData[i])
    return updatedValues # it returns an updated weight data for winner part
