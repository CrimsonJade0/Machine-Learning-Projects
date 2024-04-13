"""
Created on Fri Nov 10 09:07:32 2023
@author: Blair Bram
"""
from sklearn.model_selection import train_test_split 
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
dataset = datasets.load_breast_cancer() 

#seperate x data 
x = dataset.data 
#seperate y target 
y = dataset.target

#80 - 20 split for train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 

# hidden layer (size, layers), solver one of 3 suggested in decription, alpha varies , random state 
MLPclassif = MLPClassifier(solver = 'lbfgs', alpha = 1.5, hidden_layer_sizes= (6,2), random_state=1, max_iter=400)
#note to avoid itteration error increase
 
#train network 
MLPclassif.fit(x_train,y_train)

#test network 
y_hat = MLPclassif.predict(x_test)
print('Actual:  Predicted:')
for y_test, y_hat in zip(y_test, y_hat):
   print(f"{y_test}          {y_hat}")
