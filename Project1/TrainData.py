import numpy as np 
import MinWeightData as WinnerVector
import UpdateWeight as Updt


def trainData(weightData, trainData, iteration, learningRate):
    index = 0
    for i in range(iteration):
        for j in range (trainData.shape[0]):
            # trainData.shape[0] : number of training data i have
            winner = WinnerVector.min(weightData, trainData[j] , index, weightData.shape[0])
            updatedVector = Updt.update(weightData[winner], trainData[j], learningRate)
            weightData[winner] = updatedVector
        learningRate = learningRate * 0.5
    return weightData # return the trained weight  

    """

    """   
