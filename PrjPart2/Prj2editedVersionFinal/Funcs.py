import numpy as np 

def update(weightData, trainingData, learningRate):
    # weightData   : parameter to pass the weightData winner vector for weighData array, weightData[winner].
    # trainingData : parameter to pass the trainingData[index], index of trainingData we are testing currently.
    # learningRate : pass the learingRate.
    
    updatedValues =  weightData + learningRate*(trainingData - weightData)
    return updatedValues # it returns an updated weight data for winner part
#==========================================================================================================

def EcuDist(trainDataVector, weightDataVector):
    # trainDataVector: pass the traing data vector.
    # weightDataVector: pass the weight data vector 
    resultSum = np.sum((trainDataVector-weightDataVector)**2)
    return resultSum    
    # it returns the Ecu. distance between two passed vectors.
#==========================================================================================================
"""
you pass this function the weigted matrix(output nodes) and the vector form input matrix that 
you want to check distance with the weighted matrix
"""
def min(weightData , trainingDataVector ):
    # weightData: pass the whole weightData array.
    # trainingDataVector: pass the trainingDataVector we are currently testing.
    # index : flag for holding the winner index vector.
    # weightDataTestVectorNum: parameter to have the number of rows in the weightData array to test with each training data vector.
    index = 0
    minValue = EcuDist(trainingDataVector,weightData[0]) # initalize the min
    #minValue = 0
    for i in range(weightData.shape[0]):
        if(EcuDist(trainingDataVector,weightData[i]) < minValue):
            minValue = EcuDist(trainingDataVector,weightData[i])
            index = i
    return index 
     
    # return the index of winner vector in the weigted matrix nural,
    # nuralsDataArray, return either 0 or 1, class I, or class II

#==================================================================================
def trainData(weightData, trainData, iteration, learningRate):
    print(trainData.shape[0])
    for i in range(iteration):
        for subj in range (trainData.shape[0]):
            # trainData.shape[0] : number of training data i have
            winner = min(weightData, trainData[subj]) # return the index of winner in the weighted data
            updatedVector = update(weightData[winner], trainData[subj], learningRate)
            weightData[winner] = updatedVector # update weight data winner with new one 
        learningRate = learningRate * 0.5
    return weightData # return the trained weight     

#==================================================================================
def Test(testData , traindWeight):
    """
        this function takes the test data and the trained weight data, 
        and then clasify each test data.
    """
    for i in range(testData.shape[0]):
        winnerTest = min(traindWeight,testData[i])
        print(' Test data:', i, 'belongs to : ', winnerTest)


