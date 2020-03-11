import numpy as np 
import EuclDistance as dis

"""
you pass this function the weigted matrix(output nodes) and the vector form input matrix that 
you want to check distance with the weighted matrix
"""
def min(weightData , trainingDataVector , index):
    # weightData: pass the whole weightData array.
    # trainingDataVector: pass the trainingDataVector we are currently testing.
    # index : flag for holding the winner index vector.
    # weightDataTestVectorNum: parameter to have the number of rows in the weightData array to test with each training data vector.

    minValue = dis.EcuDist(trainingDataVector,weightData[0]) # initalize the min
    #minValue = 0
    for i in range(weightData.shape[0]):
        if(dis.EcuDist(trainingDataVector,weightData[i]) < minValue):
            minValue = dis.EcuDist(trainingDataVector,weightData[i])
            index = i
    return index 
     
    # return the index of winner vector in the weigted matrix nural,
    # nuralsDataArray, return either 0 or 1, class I, or class II


