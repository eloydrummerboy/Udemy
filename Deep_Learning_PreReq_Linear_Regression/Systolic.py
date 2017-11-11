# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:54:31 2017

@author: Scott
"""

import os
os.chdir("C:\\Users\\Scott\\Documents\\Code\\Udemy\\Deep Learning PreReq - Linear Regression")



# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure <- What we want to predict, our Y
# X2 = age in years <- First Variable
# X3 = weight in pounds <- Second Variable


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#Explore with different data types
# Data Frame
data = pd.read_excel('mlr02.xls')
# Numpy Array
data_array = np.array(data)
# Numpy Matrix
data_matrix = data.as_matrix()

#Note that the last two are really the same
print(type(data_array))
print(type(data_matrix))

# Let's plot the data to verify everything
# looks good.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_array[:,0], data_array[:,1], data_array[:,2])
plt.show()

# Let's look at just Systolic vs Weight
# then Systolic vs Age
fig1 = plt.scatter(data_array[:,2], data_array[:,0])
plt.title("Systolic vs Weight")
plt.show()

fig1 = plt.scatter(data_array[:,1], data_array[:,0])
plt.title("Systolic vs Age")
plt.show()

# We can see a positive coorelation between Systolic Blood
# Pressure and both Weight and Age.

# Add our ones to act as the bias
data['ones'] = 1
 


# Going to do 3 different regressions:
    # 1) 1-dimensional with only X2 (age) as input
    # 2) 1-dimensional with only X3 (weight) as input
    # 3) 2-dimensional with all variables as input
# And compare the R-Squared values


# 1 #######################################
# Apply the Equations we learned for regression to
# calculate a and b
Y = data['X1']
X2 = data['X2']
denominator = X2.dot(X2) - X2.mean() * X2.sum()
a = ( X2.dot(Y) - Y.mean()*X2.sum() ) / denominator
b = ( Y.mean() * X2.dot(X2) - X2.mean() * X2.dot(Y) ) / denominator
# Calculate predicted Y
Yhat2 = a*X2 + b
# Calculate Rsquared
d1 = Y - Yhat2
d2 = Y - Y.mean()
r21 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-Squared 1: ", r21)

# 2 #######################################
Y = data['X1']
X3 = data['X3']
denominator = X3.dot(X3) - X3.mean() * X3.sum()
a = ( X3.dot(Y) - Y.mean()*X3.sum() ) / denominator
b = ( Y.mean() * X3.dot(X3) - X3.mean() * X3.dot(Y) ) / denominator
# Calculate predicted Y
Yhat3 = a*X3 + b
# Calculate Rsquared
d1 = Y - Yhat3
d2 = Y - Y.mean()
r22 = 1 - d1.dot(d1) / d2.dot(d2)
print("R-Squared 2: ", r22)

# 3 #######################################
Y = data['X1']
X = data[['X2','X3','ones']] 
# Calculate predicted Y
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X,w)
# Calculate Rsquared
d1 = Y - Yhat
d2 = Y - Y.mean()
r23 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-Squared 3: ", r23)



# Better yet, we could have just made a function
# and note that the Multi Dimensional solution will work for 
# single dimensions as well, we just need to include the 'ones'

def getR2(X,Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Yhat = np.dot(X,w)
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

X2 = data[['X2', 'ones']]
X3 = data[['X3', 'ones']]

print("X2 Only: ", getR2(X2,Y))
print("X3 Only: ", getR2(X3,Y))
print("Both inputs: ", getR2(X,Y))

