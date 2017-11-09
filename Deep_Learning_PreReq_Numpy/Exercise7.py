# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:08:50 2017

@author: eloy
"""

# Exercise 7
# Generate data and plot "Donut" a.k.a. "Concentric Circles"
# Equation for a circle in euclidean plane:
# y = sin(x) + cos(x)

import numpy as np
import matplotlib.pyplot as plt



def get_circle_coordinates(radius,N):    
    hemisphere = (2*np.random.binomial(1,0.5,N))-1
    noise = np.sqrt(radius)*np.random.randn(N)/3
    #x = np.linspace(-radius,radius,N) 
    #y = np.sqrt(radius**2 - x**2)*hemisphere+noise
    x = np.random.uniform(-radius,radius,N)+noise
    y = np.sqrt(radius**2 - x**2)*hemisphere+noise
    return x,y

radius = 20
N = 50*radius

circle1_x, circle1_y = get_circle_coordinates(radius,N)


radius = 10
N = 50*radius

circle2_x, circle2_y = get_circle_coordinates(radius,N)

plt.scatter(circle1_x,circle1_y,c='red',s=40, alpha=.4)
plt.scatter(circle2_x,circle2_y,c='blue',s=40, alpha=.4)
plt.axis('equal')
plt.show()

def parameterization_get_coord(radius,N):
    t = np.linspace(0,4*np.pi,N)
    noise = np.random.randn(N)
    x = radius*np.cos(t)+noise
    y = radius*np.sin(t)+noise    
    return x,y
    
radius = 20
N = 10*radius

c1_x, c1_y = parameterization_get_coord(radius,N)


radius = 10
N = 10*radius

c2_x, c2_y = parameterization_get_coord(radius,N)

plt.scatter(c1_x,c1_y,c='red',s=40, alpha=.4)
plt.scatter(c2_x,c2_y,c='blue',s=40, alpha=.4)
plt.axis('equal')
plt.show()
    