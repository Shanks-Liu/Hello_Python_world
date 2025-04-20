# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:48:44 2019

@author: Shanks
"""

import numpy as np

data = np.array([13.3, 6.2, 15.5, 16.3, 13.6, 7.3, 5, 6.4, 4.777])
data2 = np.array([6.579, 4.881, 12.629, 15.396, 13.578, 7.262, 4.594, 5.577, 4.456])
#print(data.shape[0])
#print(data.size)
#beta = int(input("give a number beta:  "))

beta = np.array([10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])
results = np.empty(data2.shape[0])

for j in range(beta.shape[0]):
    for i in range(data2.shape[0]):
        results[i] = np.power(np.average(np.power(data2[i:].copy(), beta[j])), 1/beta[j])
    print(np.round(results, 2))

#print(data)