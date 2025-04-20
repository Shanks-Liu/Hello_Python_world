# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:21:23 2018

@author: 游侠-Speed
"""

import matplotlib.pyplot as plt

x_values = list(range(1, 5001))
cubes = [x**3 for x in x_values]
plt.scatter(x_values, cubes, c= x_values, cmap = plt.cm.Reds, s = 12)
plt.title("CUBE", fontsize = 24)
plt.xlabel("num", fontsize = 14)
plt.ylabel("cube", fontsize = 14)
plt.tick_params(axis = "both", labelsize = 14)
plt.axis([0, 5000, 0, 125000000000])
plt.show()
