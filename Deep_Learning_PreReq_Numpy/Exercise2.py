# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:14:12 2017

@author: eloy
"""

# Demonstrate the Central Limit Theorem
# CLT => if Y = X1 + X2 + X3 + ... + Xn
# Where X are IID (Independent and Identically Distributed)
# Then as N => Infinity, Y => Gaussian Normal Distribution

# We will use the Uniform Dist as our base for X
# Use N = 1000 to start
# Then draw 1000 Y's 
# Plot histogram of Y's (Should be "bell curve" if following Norm Dist)

import numpy as np
import matplotlib.pyplot as plt

Nx = 5000
Ny = 5000

Y = []

for x in range(1,Ny+1):
    
    X = np.random.uniform(0,1,Nx)
    Y.append(np.sum(X))
    
plt.hist(Y,bins=100)
plt.show()


# Expected mean: we have Nx numbers, uniformly distributed
# from 0 to 1 all summed up. The average of X should be 0.5, so the
# mean of Y should be Nx*0.5
expected_mean = Nx*0.5
expected_variance = 

Ynp = np.array(Y)
print("Expected mean of Y: ", expected_mean )
print("Mean of Y: ", Ynp.mean())
