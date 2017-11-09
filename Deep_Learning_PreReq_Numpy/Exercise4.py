# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 18:29:35 2017

@author: eloy
"""

# Write a function that flips an image 90 degrees clockwise

import os
os.chdir("/home/eloy/Data_Science/Kaggle/MNIST/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

test = df.loc[df['label'] == 9]
test_mu = test.mean()
test_mu = np.array(test_mu)
test_mu = test_mu[1:]
test_mu = test_mu.reshape(28,28)


plt.imshow(test_mu)
plt.show()




# Rotate with For Loops
# rotated[0,0] will == test_mu[0,27]
# rotated[0,1] will == test_mu[1,27]
# rotated[1,0] will == test_mu[0,26]
# so rotated row = 27 -test_mu col
# and rotated col = 27 - test_mu row
# a.k.a. rows become columns, columns are reversed
rotated_for = np.ones((28,28))
for row in range(0,len(test_mu)):
    for col in range(0,len(test_mu)):
        rotated_for[row,col] = test_mu[27-col,row]
        
plt.imshow(rotated_for)
plt.show()
    
# Rotate with numpy
# Rows become Columns
rotated_np = test_mu.transpose()
# Columns are reversed
rotated_np = rotated_np[:,::-1]
plt.imshow(rotated_np)
plt.show()    