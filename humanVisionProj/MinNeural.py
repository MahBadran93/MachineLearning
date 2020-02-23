import numpy as np 
import EuclDistance as dis

"""
you pass this function the weigted matrix(output nodes) and the vector form input matrix that 
you want to check distance with the weighted matrix
"""
def min(nuralsDataArray , inputVector , index):
    minValue = dis.EcuDist(inputVector,nuralsDataArray[0]) # initalize the min
    for i in range(2):
        if(dis.EcuDist(inputVector,nuralsDataArray[i])<minValue):
            minValue = dis.EcuDist(inputVector,nuralsDataArray[i])
            index = i
    return index  # return the index of winner vector in the weigted matrix nural,
                               # nuralsDataArray


