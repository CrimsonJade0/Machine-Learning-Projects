"""
Created on Fri Sep 22 08:56:48 2023

@author: Blair Bram
"""
import numpy as np
x = [3.10, 4.0, 3.91, 2.94, 3.0, 3.52, 3.63, 3.76, 3.46, 3.23] # GPA data 


def normalize(x): #normalizataion function 
    min_x = np.min(x)
    max_x = np.max(x)
    normalized_data = [round((val - min_x) / (max_x - min_x),3) for val in x]
    return normalized_data

normalized_data = sorted(normalize(x))
print(f"Normalization Method: {normalized_data}")

def standar(x): #standardizedion function 
    u = np.mean(x) 
    o = np.std(x)
    standard_val = [round(((val - u) / o), 3) for val in x]
    return standard_val

standard_val = sorted(standar(x))
print(f"Standardizedion Method: {standard_val}")