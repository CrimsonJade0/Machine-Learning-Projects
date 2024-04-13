"""
Created on Fri Dec  1 09:06:42 2023
@author: Blair Bram
""" 
import numpy as np
from sklearn import datasets
dataset = datasets.load_diabetes() 

#seperate x data 
x = dataset.data 
#seperate y target (dont need but good check) 
y = dataset.target

#dimensions 
[n,m] = x.shape
#find mean of all samples 
mu = np.mean(x, axis=1, keepdims=True)

#number of features - 1 (10 feats total)
k = n - 6
#adjust the values for a zero mean (4
x = x - mu

#Covariance matrix (5) 
Cov = 0 
for i in range(m):
    x_vec = x[:,i].reshape(n,1)
    Cov = Cov + 1/m * np.dot(x_vec,x_vec.T)

#Finding the SVD matrix U 
[U, S, V] = np.linalg.svd(Cov)

#Reduce U 
Ured = U[:, 0:k]

#Obtain the PCA dataset 
Z = np.dot(Ured.T, x)

# Reconstruct data
X_reconstructed = np.dot(Ured, Z)

# Calculate  error matrix
error_matrix = np.abs(x - X_reconstructed)

# Calculate percentage error matrix
perc_error_matrix = (error_matrix / np.abs(x)) * 100

# Calculate average absolute error
avg_error = np.mean(perc_error_matrix)

