import numpy as np 
import matplotlib.pyplot as plt

arrayt = [
    [1,2,3],
    [5,6,9],
    [3,3,3],
    [2,1,5]
]
arr = np.array([1,2,3,4,5])
print('shape1',arr.shape)
print(arr)

ff = arr.reshape(5,1)
print('shape2',ff.shape)
print(ff)

data = np.array(arrayt)

test1,test2 = 5 ,2
print(np.shape(data) , test2)