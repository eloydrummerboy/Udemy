# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:37:28 2017

@author: eloy
"""

import numpy as np
from datetime import datetime

a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

def slow_dot_product(a,b):
    result = 0
    for e, f in zip(a,b):
        result += e*f
    return result
    
t0 = datetime.now()
for t in range(T):
    slow_dot_product(a,b)

dt1 = datetime.now() - t0

t0 = datetime.now()
for t in range(T):
    a.dot(b)
dt2 = datetime.now() - t0

print("dt1: ", dt1.total_seconds(),"s")
print("dt2: ", dt2.total_seconds(),"s")
print("ratio: ", dt1/dt2)