# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
#-----List and Arrays----------------------------------------
L_hom= [1, 2, 3]

L_het= [1, 'a', 3.14, "Hello world."]

L_het.append("new item")

L_het[1]= 'b'

for i in range(len(L_hom)):  #
    L_hom[i] = 2 * L_hom[i]
    
L_hom = 2*L_hom  

Arry_list = np.reshape(L_hom,(3,2)) # reconstructs array to 
#-----Dictionaries---------------------------------------------
vehicle = {"Owner": "Khorbotly", "Company": "Toyota", 
           "Model": "Camry", "Color": "Black", "Year": 2014}
print(vehicle)
print(vehicle["Color"])

vehicle["Color"] = "Red"

#---Functions---------------------------------
def Stats(x):
    Valid = True
    if not isinstance(x,list):
        Valid = False 
        print("The input argument is not a list")
    else :
        for i in x: 
            if type(i)!=int and type(1)!= float:
                print("Stats of these numbers can not be obtained for they must be numerical values")
                Valid = False
                break
    if Valid:
        mean = np.mean(x)
        median = np.median(x)
        std_dev = np.std(x)
        maximum = np.max(x)
        minimum = np.min(x)
        length= len(x)
        stats = {"Mean": mean, "Median": median, "std_dev": std_dev, "Maximum": maximum, "Minimum": minimum, "Length": length} 
        print(stats)
    else: 
        mean = np.nan
        median = np.nan
        std_dev = np.nan
        maximum = np.nan
        minimum = np.nan
        length = "n/a"
        return mean,median,std_dev,maximum,minimum,length