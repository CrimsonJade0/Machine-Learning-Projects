# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:41:14 2023
@author: Blair Bram
"""
''' DO NOT NEED GENDER OR BIO SEX SINCE ALL INDIVIDUALS ARE WOMAN 
Notes: 
   SYMP_1:Feeling failure, SYMP_2:Difficulty concentrating, SYMP_3:Low energy, SYMP_4:Moving slowly
   SYMP_5:Sleep difficulties, SYMP_6:Appetite changes, SYMP_7:Anhedonia, SYMP_8:Low mood, SYMP_9:Suicidal ideation.
  PREV_MHTX:Previous Mental Health Treatment
  
'''
#implimentation of niave bisiaen 
import pandas as pd

data = pd.read_csv('depression_network_data.csv')
#get rid of nan data 

#bin race between black and white 