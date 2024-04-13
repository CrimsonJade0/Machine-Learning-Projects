"""
Created on Fri Oct  6 09:05:34 2023
@author: Blair Bram
"""
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import pandas as pd 
from bokeh.plotting import figure 
from bokeh.io import show, output_file
data = pd.read_csv('Moive list.csv')
#feats chosen is genra and year 
#got rid of unneeded columns 
data = data.drop(["Genre ", "Rating"], axis='columns')

def scale(x,l,h):
    re = (x-l)/(h-l)
    return re

ll=data['Length (Mins)'].min()
hl=data['Length (Mins)'].max()
data['Scaled_Length (Mins)']=data['Length (Mins)'].apply(scale,args=(ll,hl))

ly=data['Published Year'].min()
hy=data['Published Year'].max()
data['Scaled_Published Year']=data['Published Year'].apply(scale,args=(ly,hy))

#made current data into 2D graph 
def color(C):
    if (C == 1):
        color = "green"
    else:
        color = 'red'
    return (color)
#apply green to 1 class and red to 0 class 
data['Color']=data['Class'].apply(color)


plot = figure(x_axis_label='Published Year', y_axis_label='Length (Mins)')
plot.circle(data['Scaled_Published Year'], data['Scaled_Length (Mins)'], color = data['Color'], size = 3)
output_file('movie_plot.html')
show(plot)

#create training data set 
X= data.copy()
Y = X['Class']
X = X.drop(['Class', 'Movie name', 'Color','Published Year','Length (Mins)'], axis=1)


k = 5
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X, Y)
#regression
neigh_r=KNeighborsRegressor(n_neighbors=k)
neigh_r.fit(X,Y)
#input a scaled testcase 
testcase = [[0.321,0.372]]

#getting scaled testcase 
testy = round((testcase[0]-ly)/(hy-ly),3)
testl = round((testcase[1]-ll)/(hl-ll),3)
print(testy)
print(testl)

p_val = neigh.predict(testcase)
p_valr = neigh_r.predict(testcase)

print('Tested Movie: Grease')
print('Origional Value: 1978,110')
print(f'Scaled Value: {testcase}')
print("Predicted Value (Classification):")
print(p_val)
print("Predicted Value (Regression):")
print(p_valr)
