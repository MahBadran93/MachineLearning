import numpy as np 

def update(weightData, trainingData, learningRate):
    # weightData   : parameter to pass the weightData winner vector for weighData array, weightData[winner].
    # trainingData : parameter to pass the trainingData[index], index of trainingData we are testing currently.
    # learningRate : pass the learingRate.

    # this function returns an updated weight data for winner part
    updatedValues =  weightData + learningRate*(trainingData - weightData)
    return updatedValues 
#==========================================================================================================

def EcuDist(trainDataVector, weightDataVector):
    # trainDataVector: pass the traing data vector.
    # weightDataVector: pass the weight data vector 

     # tbis function returns the Ecu. distance between two passed vectors
    resultSum = np.sum((trainDataVector-weightDataVector)**2)
    return resultSum    
   
#==========================================================================================================

def min(weightData , dataSetVector ):
    # weightData: pass the whole weightData array.
    # dataSetVector: pass the a data vector to compare with weightdata array, to find its closest match 
    """
     this function returns the index of winner in the weigted matrix(closest match with trainingdata vector). result is 0 ro 1
    """
    winnerIndex = 0 #flag for initalizing the winner index
    minValue = EcuDist(dataSetVector,weightData[0]) # initalize the minValue
    # iterate through all weighdata rows to find the closest match, depending on ecu. distance,
    #and then return the index of the closest match(winner)
    for i in range(weightData.shape[0]):
        if(EcuDist(dataSetVector,weightData[i]) < minValue):
            minValue = EcuDist(dataSetVector,weightData[i])
            winnerIndex = i
    return winnerIndex 
     
   

#==================================================================================
def trainData(weightData, trainData, iteration, learningRate):
    for i in range(iteration):
        for subj in range (trainData.shape[0]):
            # trainData.shape[0] : number of training data(subjects) I have
            winner = min(weightData, trainData[subj]) #return index of winner to update accordingly
            updatedVector = update(weightData[winner], trainData[subj], learningRate) # return updated values
            weightData[winner] = updatedVector #replace the winner from  the weightdata with updated one
        learningRate = learningRate * 0.5 #change learning Rate each iteration 
    return weightData # return the final trained weight     

#==================================================================================