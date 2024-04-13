"""
Created on Mon Oct 23 09:12:52 2023
@author: Blair Bram
"""
import pandas as pd 
import numpy as np
import math 

data = pd.read_csv('Social_Network_Ads.csv')

# Bin Gender 
data['G'] = data['Gender'].apply(lambda x: 0 if x == 'Male' else 1) 

# Normalize age and salary (create new columns)
minA = data.Age.min()
minES = data.EstimatedSalary.min()
MaxA = data.Age.max()
MaxES = data.EstimatedSalary.max()
data['nA'] = data['Age'].apply(lambda x: (x - minA) / (MaxA - minA))
data['nES'] = data['EstimatedSalary'].apply(lambda x: (x - minES) / (MaxES - minES))

# Make List and stack x(gender, age, salary) 3x400 
G = data['G'].values
Nage = data['nA'].values
Nes = data['nES'].values

X = np.vstack((G, Nage, Nes))  # Matrix of inputs 
Purch = data['Purchased'].values  # vectorize y 1x400

# Make initial values for w0, w1, w2, b, and alpha 
wT = np.array([0.39, 0.49, 0.65])
b = 4
alpha = 5

m = X.shape[1]
print('Iteration\tCost Function\t      w\t                   b')

for c in range(100):
    J = 0
    dw = np.zeros([3, 1])
    db = 0

    for i in range(m): 
        X_i = X[:, i].reshape(3, 1)
        z = np.dot(wT, X_i) + b  
        y_hat = 1 / (1 + math.exp(-z)) 
        J -= (Purch[i] * math.log10(y_hat) + (1 - Purch[i]) * math.log10(1 - y_hat))
        e = y_hat - Purch[i] 
        dw += X_i * e
        db += e 

    J /= m
    dw /= m
    db /= m
    wT = wT - alpha * dw.flatten()  
    b -= alpha * db

    print(f'{c}\t\t {round(J, 4)}\t\t {[round(val, 4) for val in wT]}\t\t {round(b, 4)}')
    #looking for cost function to be as  close to 0 as possiable 
print(f'Final values: Cost = {J}, w = {wT}, b = {b}')


# Testing Code function
def predict(input_data, weights, bias):
    z = np.dot(weights, input_data) + bias
    return 1 / (1 + math.exp(-z))

X_test = np.array([0, 0.78, 0.97])  

# Use the trained weights and bias for predictions
y_hat_test = predict(X_test, wT, b)
class_prediction = round(y_hat_test)

print(f'Test Case Prediction: {class_prediction}')



