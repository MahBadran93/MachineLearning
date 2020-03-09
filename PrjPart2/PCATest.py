import numpy as np
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cv2 as cv



np.convolve
data = np.genfromtxt('./predicting-a-pulsar-star/pulsar_stars.csv' , delimiter=',', dtype='float64')

scalar = MinMaxScaler(feature_range=[0,1])
data_normalized = scalar.fit_transform(data[1: , 0:8])


#Fitting the PCA algorithm with our Data
pca = PCA().fit(data_normalized)#Plotting the Cumulative Summation of the Explained Variance
pca = PCA(n_components=5)
dataset = pca.fit_transform(data_normalized)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()
"""
dataMean = np.mean(data_normalized)
data_center = np.abs(data_normalized - dataMean)
ss = data_center.T
covM = np.cov(ss)

eigenval, eigenVec = np.linalg.eig(covM)

significance = [np.abs(i)/np.sum(eigenval) for i in eigenval]
print(eigenVec)
"""