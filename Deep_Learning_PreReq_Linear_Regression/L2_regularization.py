# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:23:47 2017

@author: Scott
"""

# L2 Regularization

import numpy as np
import matplotlib.pyplot as plt

# Set Number of Data Points
N = 50

X = np.linspace(0,10,N)
Y = 0.5*X + np.random.randn(N)

# Manually creat outliers by setting the last two points to
# 30 greater than they were
Y[-1] += 30
Y[-2] += 30
 
# Plot the data to ensure it looks as expected
plt.scatter(X,Y)
plt.show()
  
# Add columns of 1s for our bias term
X = np.vstack([np.ones(N), X]).T

             
# Calculate the maximum-liklihood solution, as before
# (will not account for outliers)
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)       

plt.scatter(X[:,1],Y)
plt.plot(X[:,1], Yhat_ml)
plt.show()


# Set L2 Penatly. This is an adjustable feature, or a "tuning parameter"
l2 = 1000
# just need to add in the penatly for large weights by
# adding in the identify matrix I * l2, which will essentially
# add l2 to each diagonal element of X.T (dot) X 
# array([[ 1000.,     0.],       array([[   50.        ,   250.        ],      
#        [    0.,  1000.]])   +         [  250.        ,  1683.67346939]])
# The 2 in the eye() funciton comes from X being 2 dimensional
# We could have also used X.shape[1]
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)       

plt.scatter(X[:,1],Y)
plt.plot(X[:,1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1], Yhat_map, label='map')
plt.legend()
plt.show()



      