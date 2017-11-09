# -*- coding: utf-8 -*-
"""
Spyder Editor

Linear Regression in Python
Udemy Course

"Coding the 1-D Solution in Python"
"""

import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('/home/eloy/Programming/machine_learning_examples-master/linear_regression_class')

# load data
X = []
Y = []
for line in open('data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# Let's turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

plt.scatter(X,Y)
plt.show()

# Apply the Equations we learned for regression to
# calculate a and b

denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

# Calculate predicted Y
YHat = a*X + b

# Plot it all
plt.scatter(X,Y)
plt.plot(X,YHat)
plt.show()

from datetime import datetime

startMeUp = datetime.now();

# R Squared (my original take)
SSres = 0
SStot = 0
Ymean = Y.mean()

for i in range(0,len(Y)):
    SSres += (Y[i] - YHat[i])**2
    SStot += (Y[i] - Ymean)**2
    
Rsq = 1 - (SSres/SStot)
endMe = datetime.now()

print("R squared = ", Rsq)
print("Time to execude = ",endMe - startMeUp)

# R Squared (Instructor's Code)
# Sum of product of two arrays is the dot product
# If you're squaring an array and summing the results, 
# That's the dot product of the array with itself

start = datetime.now();
d1 = Y - YHat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
end = datetime.now()
print("Instructor's R Squared = ", Rsq)
print("Time to execude = ",end - start)
