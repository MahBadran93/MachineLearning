import numpy as np 

# this function returns an updated weight data for winner part
def update(weightData, trainingData, learningRate):
    # weightData   : parameter to pass the weightData winner vector for weighData array, weightData[winner].
    # trainingData : parameter to pass the trainingData[index], index of trainingData we are testing currently.
    # learningRate : pass the learingRate.
    updatedValues =  weightData + learningRate*(trainingData - weightData)
    return updatedValues 
#==========================================================================================================
# this function returns the Ecu. distance between two passed vectors
def EcuDist(trainDataVector, weightDataVector):
    # trainDataVector: pass the traing data vector.
    # weightDataVector: pass the weight data vector 
    resultSum = np.sum((trainDataVector-weightDataVector)**2)
    return resultSum    
#==========================================================================================================
"""
this function returns the index of winner in the weigted matrix(closest match with dataSetVector vector),
result is 0 or 1
"""
def min(weightData , dataSetVector ):
    # weightData: pass the whole weightData array.
    # dataSetVector: pass the a data vector to compare with weightdata array, to find its closest match 
    winnerIndex = 0
    minValue = EcuDist(dataSetVector,weightData[0]) # initalize the min
    #minValue = 0
    """
    iterate through all weighdata rows to find the closest match, depending on ecu. distance,
    and then return the index of the closest match(winner)
    """
    for i in range(weightData.shape[0]):
        if(EcuDist(dataSetVector,weightData[i]) < minValue):
            minValue = EcuDist(dataSetVector,weightData[i])
            winnerIndex = i
    return winnerIndex 
     
#==================================================================================
"""
This function returns a final trained weight data 

"""
def trainData(weightData, trainData, iteration, learningRate):
    for i in range(iteration):
        for subj in range (trainData.shape[0]):
            # trainData.shape[0] : number of training data(subjects) in the training data
            winner = min(weightData, trainData[subj]) #return index of winner to update accordingly
            updatedVector = update(weightData[winner], trainData[subj], learningRate) # return updated values
            weightData[winner] = updatedVector #replace the winner from  the weightdata with updated one 
        learningRate = learningRate * 0.5
    return weightData # return the trained weight     

#==================================================================================
def Test(testData , traindWeight):
    """
        this function takes the test data and the trained weight data, 
        and then clasify each test data.
    """
    for i in range(testData.shape[0]):
        # testData.shape[0] : number of training data(subjects) in the testing data
        winnerTest = min(traindWeight,testData[i])
        if(winnerTest == 0):
            winnerTestResult = 'patient'
        elif(winnerTest == 1):
            winnerTestResult = 'control'
        print(' Test data:', i, 'belongs to : ', winnerTestResult)


