# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:16:54 2017

## Udemy - Lazy Programmer - Deep Learning Prerequisite
## Linear Regression in Python
## Section 3 Lecture 16
## Polynomial Regression

@author: Scott
"""

import os
os.chdir("C:\\Users\\Scott\\Documents\\Code\\Udemy\\Deep Learning PreReq - Linear Regression")

import numpy as np
import matplotlib.pyplot as plt


# We can use the methods and techniques from linear Regression
# on polynomials, because 'linear' refers to the weights, not
# necessarily the variables

X = []
Y = []

for line in open('data_poly.csv'):
    x,y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))
    
# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# let's plot to check out the data
plt.scatter(X[:,1], Y)
plt.show()
    
# Calculate our weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X,w)

# plot it all together
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.show()

# R-squared
# Compute R-squared to see how good our
# model is.
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-Squared: ", r2)
