## Udemy - Lazy Programmer - Deep Learning Prerequisite
## Linear Regression in Python
## Section 3 Lecture 15
## Mult-Dimensional Solution

import os
os.chdir("C:\\Users\\Scott\\Documents\\Code\\Udemy\\Deep Learning PreReq - Linear Regression")

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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

# Let's plot the data to verify everything
# looks good.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()
    
# Calculate weights using equations from previous
# lectures
w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T, Y))
Yhat = np.dot(X,w)

# Compute R-squared to see how good our
# model is.
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-Squared: ", r2)

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
ax.scatter(X[:,0], X[:,1], Yhat)
plt.show()
#END