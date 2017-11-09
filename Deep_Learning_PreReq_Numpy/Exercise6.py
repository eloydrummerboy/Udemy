# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 22:40:13 2017

@author: eloy
"""

# Exercise 6: Generate the XOR dataset and plot it
# i.e. 4 quadrants of random data -1 to 1 in 2D
# >0 = True, <0 = False
# quadrant 1 and 3 are colored blue,
# quadrant 1, x>0, y>0, XOR(True, True) = True
# quadrant 3, x<0, y<0, XOR(False, False) = True
# and similar for Quadrants 2 and 4, except they're
# colored red

import numpy as np
import matplotlib.pyplot as plt

data = np.random.uniform(-1,1,size=(3000,2))
blue_data = data[((data[:,0]>0) & (data[:,1]>0)) | ((data[:,0]<0) & (data[:,1]<0))]
red_data = data[((data[:,0]>0) & (data[:,1]<0)) | ((data[:,0]<0) & (data[:,1]>0))]

plt.scatter(blue_data[:,0],blue_data[:,1],c='b',alpha = .4)
plt.scatter(red_data[:,0],red_data[:,1],c='r',alpha = .4)
plt.show()

