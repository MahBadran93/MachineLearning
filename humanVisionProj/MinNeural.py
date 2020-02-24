import numpy as np 
import EuclDistance as dis

"""
you pass this function the weigted matrix(output nodes) and the vector form input matrix that 
you want to check distance with the weighted matrix
"""
def min(nuralsDataArray , inputVector , index):
    minValue = dis.EcuDist(inputVector,nuralsDataArray[0]) # initalize the min
    for i in range(2):
        #print('D' , i , dis.EcuDist(inputVector,nuralsDataArray[i] ))
        if(dis.EcuDist(inputVector,nuralsDataArray[i]) < minValue):
            minValue = dis.EcuDist(inputVector,nuralsDataArray[i])
            index = i
            #print('weight' , i , minValue )
    return index  # return the index of winner vector in the weigted matrix nural,
                               # nuralsDataArray, return either 0 or 1, class I, or class II


