# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:56:34 2017

@author: eloy
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('/home/eloy/Programming/machine_learning_examples-master/linear_regression_class')


X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')
    
    x = int(non_decimal.sub("",r[2].split('[')[0]))
    y = int(non_decimal.sub("",r[1].split('[')[0]))
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X,Y)
plt.title("Y")
plt.show()

# Take log of Y to transform exponential non-linear
# problem into linear problem
Y = np.log(Y)
plt.scatter(X,Y)
plt.title("logY")
plt.show()


denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

YHat = a*X + b

plt.scatter(X,Y)
plt.plot(X,YHat)
plt.title("Line of Best Fit")
plt.show()

#How far off are we from actual values?
d1 = Y - YHat 
#How far off would we be had we just predicted the mean?
d2 = Y - Y.mean() 

Rsq = 1 - d1.dot(d1) / d2.dot(d2)
print("R squared = ", Rsq)


# Calculate how long it takes to double
# Y = log(trans. count) = a*year + b
# transistor count (tc) = e^(a*year + b) = exp(a*year) * exp(b)
# So what's the double of the tc look like?
# 2*tc = 2*exp(a*year)*exp(b) = exp(ln(2)) * exp(a*year)*exp(b)
#      = exp(a*year+ln(2) * exp(b)
# exp(b)*exp(a*year2) = exp(b)*exp(a*year1 + ln2)
# a*year2 = a*year1 + ln2
# year2 = year1 + ln2/a
print("time to double: ", np.log(2)/a," years.")




