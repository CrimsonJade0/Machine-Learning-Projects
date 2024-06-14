# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:25:14 2023
--------------------------------------------------------------------
A first attempt at a neural network that takes in a csv list that is full of Social Network data 
The full program bins the catagorical info, and trains the model to perdict what a set of data would like to see 
--------------------------------------------------------------------
@author: Blair Bram
"""
import pandas as pd 
import numpy as np

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

def int_param(ni, nh, no): #inital parameters 
    np.random.seed(2) 
    w1 = np.random.randn(nh, ni) * 0.01
    b1 = np.zeros((nh, 1))
    w2 = np.random.randn(nh, nh) * 0.01
    b2 = np.zeros((no, 1))
   
    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2}
    return parameters 

def forward_propagate(X, parameters):
       #get dictonary of parameters  
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']
      
        #foward propgating 
        z1 = np.dot(w1, X)+b1
        A1 = np.tanh(z1)
        z2 = np.dot(w2, A1)+b2
        A2 = 1/(1+np.exp(-z2)) 
        
        cache = {'z1': z1,
                'A1': A1,
                'z2': z2,
                'A2': A2} 
        return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1] # number of examples 
    
    #calculating Loss 
    logprob = np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))
    #calculting Cost 
    cost = -1/m*np.sum(logprob)
    #change cost from a vector to a scaler  
    cost = float(np.squeeze(cost))
    return cost

def back_prop(parameters, cache, X, Y):
    
    m = Y.shape[1]
    
    #get weights 1 & 2 dictionaries and get A 1 & 2 dictionaries 
    w1 = parameters['w1']
    w2 = parameters['w2']
    A1 = cache['A1']
    A2 = cache['A2']
    
    #do backwards propagation calcuations to get 
    dz2 = A2-Y
    db2 = 1/m*np.sum(dz2, axis=1, keepdims = True)
    dw2 = 1/m*np.dot(dz2,A1.T)
    dz1 = np.dot(w2.T,dz2)*(1-np.power(A1,2))
    db1 = 1/m*np.sum(dz1, axis=1, keepdims = True)
    dw1 = 1/m*np.dot(dz1,X.T)
    
    fin_param = {'dw1': dw1,
                 'db1': db1,
                 'dw2': dw2,
                 'db2': db2}
    return fin_param

def update_param(parameters, fin_param, learning_rate = 0.1):
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']
        
        dw1 = fin_param['dw1']
        db1 = fin_param['db1']
        dw2 = fin_param['dw2']
        db2 = fin_param['db2']
        
        w1 = w1 - learning_rate*dw1
        b1 = b1 - learning_rate*db1
        w2 = w2 - learning_rate*dw2
        b2 = b2 - learning_rate*db2
        
        parameters = {'w1': w1,
                      'b1': b1,
                      'w2': w2,
                      'b2': b2}
        return parameters
#set ni nh and no 
ni = 3
nh = 1
no = 1   
    # call int_param to initialize the parameters 
parameters = int_param(ni, nh, no)
    # a for loop that will run for L times    
L = 1100;
alpha = 0.6;
for i in range (L):
    #Forward Propagation 
    A2, cache = forward_propagate(X, parameters)
    
    #Compute Cost
    cost = compute_cost(A2, Purch.reshape(1, -1))
    
    #Backward Propagation
    fin_param = back_prop(parameters, cache, X, Purch.reshape(1, -1))
    
    #Update Parameters lower than 
    parameters = update_param(parameters, fin_param, alpha)
    
    print(f'Iteration {i}\tCost: {round(cost, 4)}')
    #looking for cost function to be as  close to 0 as possiable 
print(" ")
#print(f'Number of hidden nodes: {nh}')
#print(f'Value for alpha: {alpha}')
#print(f'Value for number of loops: {L}')
print(f'Final values: Cost = {cost}')

#where x is the training sample and parameters is the dictionary 
def predict(X, parameters):
    A2, _ = forward_propagate(X.T, parameters)  # Transpose X before forwarding
    predictions = (A2 > 0.5).astype(int)
    return predictions

trained_parameters = parameters  #training loop updates the parameters
new_sample = np.array([[0.1, 0.8, 0.9]])  # Replace with your actual new sample

#make a prediction 
prediction = predict(new_sample, trained_parameters)
print(f'New sample variables: {new_sample}')
print(f'Prediction for the new sample: {prediction}')



    
