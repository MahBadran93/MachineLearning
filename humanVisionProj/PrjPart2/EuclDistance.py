import numpy as np 

def EcuDist(trainDataVector, weightDataVector):
    # trainDataVector: pass the traing data vector.
    # weightDataVector: pass the weight data vector 
    sum = resultSum = 0
    for i in range(len(trainDataVector)):
        sum = sum + (trainDataVector[i]-weightDataVector[i])**2
        resultSum = np.sqrt(sum)
    return resultSum    
    # it returns the Ecu. distance between two passed vectors.