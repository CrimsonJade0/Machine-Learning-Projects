# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:30:43 2023

@author: Blair Bram
"""
import numpy as np

# Define the objective function
def objective_function(x, y, z):
    return x**2 - y * np.sin(z) + z * np.exp(-0.5 * x)

# Initialize parameters
x = -1.001  # Close to zero
y = -1.00001  # Close to zero
z = -0.009  # Close to zero

# Learning rate
alpha = 0.001  # Start with a small learning rate

# Number of iterations
iterations = 1000  # You may need more iterations for better convergence

# Training loop
for i in range(iterations):
    grad_x = 2 * (2 * x - y * np.sin(z) + z * np.exp(-0.5 * x))
    grad_y = 2 * (-np.sin(z))
    grad_z = 2 * (-y * (np.cos(z) + np.exp(-0.5 * x)))

    x -= alpha * grad_x
    y -= alpha * grad_y
    z -= alpha * grad_z

# Print the final values
print(f"Final values: x = {x}, y = {y}, z = {z}")
print(f"Objective function value (P) at the end: {objective_function(x, y, z)}")
