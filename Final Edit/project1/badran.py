#...........Result......................
"""
Test Result:

testData: [0.  0.  0.  0.9] belongs to class  1
testData: [0.  0.  0.8 0.9] belongs to class  1
testData: [0.7 0.  0.  0. ] belongs to class  2
testData: [0.7 0.9 0.  0. ] belongs to class  2

note: when you run, you can see also the trained wight data

"""
#..............Student name..............
"""
Mahmoud Badran
Project 1
"""
#.............imported libraries...................
import numpy as np 
import matplotlib.pyplot as plt 
import Funcs as F


#....................student name................

#.............................Create Data ...................................
testingData = [
    [0,0,0,0.9],
    [0,0,0.8,0.9],
    [0.7,0,0,0],
    [0.7,0.9,0,0]
]
trainingData = [
    [1,1,0,0],
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,1]
]
trainData = np.array(trainingData) # make training data as numpy array
testData = np.array(testingData) # make test data as numpy array
weightData = np.array([[.2, .6, .5, .9],[.8, .4, .7, .3]]) # create a weighdata array
#weightData = np.random.rand()

print("weight before trained:", weightData)

#........................Training........................
learningRate = 0.6
iteration = 100
# give the train data function a weight matrix and a train data to process 
weightData = F.trainData(weightData,trainData,iteration,learningRate)

print("trained weight:", weightData)

#.......................Testing...............................

print("Result.................")

def test():
    for t in range(testData.shape[0]): # iterate through all rows in test data
        winnerTest = F.min(weightData,testData[t]) # find the winner 
        print('testData:', testData[t],'belongs to class ' ,winnerTest+1)        
       
test()