# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 08:35:15 2023

@author: Blair Bram
"""
import numpy as np
import matplotlib.pyplot as plt
#import data from txt file into python 
data = np.loadtxt('food_truck_data.txt', delimiter = ",", skiprows= 1)
m = data.shape[0] #setting m to the number of sample in the file (rows)
#get data from a 97, 2 array into variables x an y 
x = data[:, 0:1]  
y = data[:, 1:2]  
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

#setting up plot with lables 
plt.scatter(x, y, marker='x', color='red') 

plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Scatter Plot of Population vs. Profit')


W = [0,0] # W starts as an empty list of 0,0
costs = [] 
alpha = 0.001
n=100
def cost(x, y, W): 
    sum_of = 0 #declaring a sum varaiable 
    for i in range(m):
        y_h = W[0]*x[i]+W[1]
        sum_of += (y[i] - y_h) **2
    return (1/(2*m))*sum_of

def update(x, y, W, alpha):
    dw1 = 0
    dw0 = 0
    for i in range(m):   
        error = y[i] - (W[0] * x[i] + W[1])
        dw1 += -2 * x[i] * error
        dw0 += -2 * error
        W[0] = W[0] - ((1 / float(m)) * alpha * dw1)
        W[1] = W[1] - ((1 / float(m)) * alpha * dw0)
    return W

def train(x, y, W, alpha, n): 
    for i in range(n): 
        W = update(x, y, W, alpha) # run update n times 
        costs.append(cost(x, y, W)) #update cost vector 
    return W 

    
trained_ws= train(x, y, W, alpha, n)
w1 = trained_ws[0]
w0 = trained_ws[1]
print(f"Our linear function is fitted to this data by: f(x) = {w1}x + {w0}")
x_regression = np.linspace(min(x), max(x), 100)
y_regression = w1 * x_regression + w0
plt.plot(x_regression, y_regression, label='Regression Line', color='blue')

plt.show()#shows line of scatter plot 

# Create a plot of cost vs. iteration number
plt.plot(range(1, n + 1), costs, marker='o')
plt.xlabel('Iteration Number')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration Number')
plt.show()
