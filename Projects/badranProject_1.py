import numpy as np 
import matplotlib.pyplot as plt 
import Funcs as F
#....................student name................
"""
Mahmoud Badran
Project 1

"""
#...........Result......................
"""
weight before trained: [[0.2 0.6 0.5 0.9],[0.8 0.4 0.7 0.3]]
after training : [[0.00897985 0.02693954 0.5885089  0.99551008],[0.99102015 0.40700118 0.03142946 0.01346977]]
in the traind weight data: 
first row, [0.00897985 0.02693954 0.5885089  0.99551008] is class 0 or class I
second row,[0.99102015 0.40700118 0.03142946 0.01346977] is class 1 or class II

Test Result:
 class: 0 testData: [0.  0.  0.  0.9] which belongs to row [0.00897985 0.02693954 0.5885089  0.99551008]
 class: 0 testData: [0.  0.  0.8 0.9] which belongs to row [0.00897985 0.02693954 0.5885089  0.99551008]
 class: 1 testData: [0.7 0.  0.  0. ] which belongs to row [0.99102015 0.40700118 0.03142946 0.01346977]
 class: 1 testData: [0.7 0.9 0.  0. ] which belongs to row [0.99102015 0.40700118 0.03142946 0.01346977]

"""

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
trainData = np.array(trainingData) # make numpy array of out input 
testData = np.array(testingData)
weightData = np.array([[.2, .6, .5, .9],[.8, .4, .7, .3]])
#weightData = np.random.rand()

print("weight before trained:", weightData)

#........................Training........................
learningRate = 0.6
iteration = 100
# give the train data function a weight matrix and a train data to process 
weightData = F.trainData(weightData,trainData,iteration,learningRate)

print("trained:", weightData)

#.......................Testing...............................

print("TESTING.................")

def test():
    for t in range(testData.shape[0]): # iterate through all rows in test data
        winnerTest = F.min(weightData,testData[t]) # find the winner 
        print(' class:', winnerTest , 'testData:', testData[t])        
       
test()