# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:30:06 2017

@author: Scott
"""

# Quiz: R2
# Does R2 improve after adding a column/feature/input of 
# random noise?

# Take our 2D data set and run linear regression like before
# to get our base R2 value
import os
os.chdir("C:\\Users\\Scott\\Documents\\Code\\Udemy\\Deep Learning PreReq - Linear Regression")

import numpy as np

# Import data
X = []
Y = []

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])
    # Remember, because of bias term, we have an X0
    # that is always 1
    Y.append(float(y))
    
# turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

# Calculate weights using equations from previous
# lectures
w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T, Y))
Yhat = np.dot(X,w)

# Compute R-squared to see how good our
# model is.
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-Squared - no noise: ", r2)

# Now add noise to X
noise = np.random.rand(len(X))
noise = noise.reshape((len(noise),1))
# Take care to make the ones column last still
#Xnoisey = np.append(X,noise,1)
Xnoisey = np.hstack((X[:,0:2], noise,X[:,2].reshape(len(X),1)))

# Solve
w = np.linalg.solve(np.dot(Xnoisey.T,Xnoisey), np.dot(Xnoisey.T, Y))
Yhat = np.dot(Xnoisey,w)

# Compute R-squared to see how good our
# model is.
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-Squared - with noise: ", r2)

# We should have actually gotten a slightly higher R squared
# but it was slightly lower. Maybe that is due to the Rsquared being so 
# high to begin with. Let's try it with the Systolic data
import pandas as pd
data = pd.read_excel('mlr02.xls')
data['ones'] = 1
    
def getR2(X,Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Yhat = np.dot(X,w)
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

Y = data['X1']
X = data[['X2','X3','ones']] 

print(getR2(X,Y))
noise = np.random.rand(len(X))
noise = noise.reshape(len(noise),1)
Xnoisey = np.append(X,noise,1)
print(getR2(Xnoisey,Y))


# This time the R Squared actually is slightly higher.
# We would expect 0 correlation between our random
# variable and our target Y. But there will almost always be
# some slight correlation. This is what improves our R squared
# value.