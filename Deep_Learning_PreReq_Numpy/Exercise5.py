# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 22:15:07 2017

@author: eloy
"""

#Excercise 5: Create a function that tests whether or not
#a matrix is symmetrical.
# Do it the "hard" way and again utilizing numpy functions

import numpy as np

    
# Manual Mode
def is_symmetric_manual(matrix):
    for row in range(0,matrix.shape[0]):
        for col in range(0,matrix.shape[1]):
            if matrix[row][col] != matrix[col][row]:
                return False
    return True            
    
 # Numpy Method
def is_symmetric_np(matrix):
    if np.array_equal(matrix,matrix.transpose()):
        return True
    else:
        return False
        
non_sym_matrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
sym_matrix = np.array([[1,2,3,4],[2,2,5,6],[3,5,7,8],[4,6,8,9]])


print(is_symmetric_manual(non_sym_matrix))
print(is_symmetric_manual(sym_matrix))
print(is_symmetric_np(non_sym_matrix))
print(is_symmetric_np(sym_matrix))        