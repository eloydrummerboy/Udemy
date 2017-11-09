# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 17:46:24 2017

@author: eloy
"""

# Plot the average of each digit in the MNIST dataset

import os
os.chdir("/home/eloy/Data_Science/Kaggle/MNIST/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

for x in range(0,10):
    temp = df.loc[df['label']==x]
    temp = np.array(temp.mean())
    temp = temp[1:]
    temp = temp.reshape(28,28)
    plt.imshow(temp,cmap='gray')
    plt.show()
    
print("end")
