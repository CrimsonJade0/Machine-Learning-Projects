"""
Created on Fri Sep 29 08:48:50 2023
@author: Blair Bram
"""

import pandas as pd 
data = pd.read_csv('Train_DataSet.csv')

#Binning Temp data loop 
data['Temp'] = ['cold' if x < 55 else ('warm' if x < 70 else "hot") for x in data['Temp']]

# make new data frame for each class for easy sorting 
On_timedf = data[data['Class'] == 'on time']
Latedf = data[data['Class'] == 'late']
V_latedf = data[data['Class'] == 'very late']
Cancdf = data[data['Class'] == 'cancelled']

#Make list for each column 
Prob = [{'On Time': '14/20', 'Late': '2/20', 'Very Late': '3/20', 'Canceled': '1/20'},
  #Day weekday, weekend, holiday
        {'On Time': '9/14', 'Late': '1/2', 'Very Late': '3/3', 'Canceled': '0/1'},
        {'On Time': '3/14', 'Late': '1/2', 'Very Late': '0/3', 'Canceled': '1/1'},
        {'On Time': '2/14', 'Late': '0/2', 'Very Late': '0/3', 'Canceled': '0/1'},
  #Temp cold, warm, hot 
        {'On Time': '6/14', 'Late': '2/2', 'Very Late': '1/3', 'Canceled': '1/1'},
        {'On Time': '5/14', 'Late': '0/2', 'Very Late': '1/3', 'Canceled': '0/1'},
        {'On Time': '3/14', 'Late': '0/2', 'Very Late': '1/3', 'Canceled': '0/1'},
  #Wind high, normal, none    
       {'On Time': '4/14', 'Late': '1/2', 'Very Late': '1/3', 'Canceled': '1/1'},
        {'On Time': '5/14', 'Late': '1/2', 'Very Late': '2/3', 'Canceled': '0/1'},
        {'On Time': '5/14', 'Late': '0/2', 'Very Late': '0/3', 'Canceled': '0/1'},
  #Rain heavy, slight, none    
        {'On Time': '1/14', 'Late': '1/2', 'Very Late': '2/3', 'Canceled': '1/1'},
        {'On Time': '8/14', 'Late': '0/2', 'Very Late': '0/3', 'Canceled': '0/1'},
        {'On Time': '5/14', 'Late': '1/2', 'Very Late': '1/3', 'Canceled': '0/1'}]
#Index rows 
ProbDF = pd.DataFrame(Prob, index=['Prior Prob',
                               'Day = Weekday',
                               'Day = Week end',
                               'Day = Holiday',
                               'Temp = Cold',
                               'Temp = Warm',
                               'Temp = Hot',
                               'Wind = High',
                               'Wind = Normal',
                               'Wind = None',
                               'Rain = Heavy',
                               'Rain = Slight',
                               'Rain = None'])


#Create new data frame for probablity table 
# columns On-time Late, Very Late, Canceled
#Then use the rows for all the other features 


#df ['new'] = [sdfsdfg if x > 10 ((all cases)) x < 20 for x in df['A]]
# function in pandas adding new colume datefram['New'] = 0 fills all col to 0
# datefram['New'] = fjgsdjfg if x > 10 else dasfsadf for x in df['A']
# in data frame A 