# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:55:59 2017

@author: eloy
"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

dist = []

A = np.array([[.3,.6,.1],
              [.5,.2,.3],
              [.4,.1,.5]])
              
v = np.array([1/float(3),1/float(3),1/float(3)])

for x in range(1,26):
    v_p = v.dot(A)
    dist.append(np.linalg.norm(v_p-v))
    v = v_p
    
#print(dist)
plt.plot(dist)
plt.show()

# since we've found v' = vA
# and v'-v after 25 runs = 0, so v' = v
# then v = vA
# This is the eigenvector for A for which the corresponding
# eigenvalue is 1