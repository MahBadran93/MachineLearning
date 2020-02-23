import numpy as np 

def EcuDist(inputVector, nuralUnitVector):
    # synaptic distance, the weight between two nodes 
    sum = resultSum = 0
    for i in range(len(inputVector)):
        sum = sum + (inputVector[i]-nuralUnitVector[i])**2
        resultSum = np.sqrt(sum)
    return resultSum    