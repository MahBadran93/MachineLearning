import numpy as np 
import matplotlib.pyplot as plt 
import EuclDistance as dis
import MinNeural as winMin
import UpdateWeight as updWeight

###########################################
inputTestData = [
    [0.7,0.9,0,0],
    [0,0,0.8,0.9],
    [0.7,0,0,0],
    [0.7,0.9,0,0]
]
inputData1 = [
    [1,1,0,0],
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,1]
]
dataInptArray = np.array(inputData1) # make numpy array of out input 
inputTest = np.array(inputTestData)

############################################
"""
below to create a random data points (map), and we will start comparing the input data
(dataInptArray) with the map data(nuralsData) for classification.
"""
# neural map 
row = 2 # map data points (neurals)
col = 4 # dimension
#nuralsData = np.random.rand(row,col) # the map with 2x4 
nuralsData = [[.2, .6, .5, .9],[.8, .4, .7, .3]]

###############################################

learningRate = 0.3
iteration = 1000
index = 0 # flag to initalize the winner vector in the weighted array  
#print("Training......",1,"Weight initalized data",nuralsData)
print("n1:", nuralsData)
for i in range(iteration):
    #print('interation......................',i,nuralsData)
    #print("learning Rate : ",learningRate)
    for j in range (4):
        #print("For Input: ",dataInptArray[j])
        winner = winMin.min(nuralsData, dataInptArray[j] , index)
        #print("winner", winner)
        updatedVector = updWeight.update(nuralsData[winner], dataInptArray[j], learningRate)
        nuralsData[winner] = updatedVector
        #print('next weight',nuralsData)
    #print("Iteration : " ,i,nuralsData)
    learningRate = learningRate * 0.5
print("n2",nuralsData)

print("TESTING.................")
def test():
    index = 0
    for t in range(len(dataInptArray)):
        winnerF = winMin.min(nuralsData, dataInptArray[t],index)
        winnerTest = winMin.min(nuralsData,inputTest[t],index)
        print(' class:', winnerF , 'input data row:', dataInptArray[t] ,"class:",winnerTest,"testData:" ,inputTest[t])        
           
test()