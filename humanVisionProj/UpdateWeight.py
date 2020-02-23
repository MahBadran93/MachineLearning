import numpy as np 


def update(nuralsData, inputData, learningRate):
    updatedValues = [0,5,0,0] # initalize
    #updatedValues =np.array([5,2,0,0])
    for i in range(len(inputData)):
        #np.copyto(updatedValues[i],nuralsData[i] + learningRate*(inputData[i] - nuralsData[i]))
        updatedValues[i] =  nuralsData[i] + learningRate*(inputData[i] - nuralsData[i])
    return updatedValues
