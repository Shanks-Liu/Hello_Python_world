# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:59:31 2018

@author: 游侠-Speed
"""

from matplotlib import pyplot as plt
from random_walk import Random

#while True:
random_exmple = Random()
random_exmple.fill_walk()
plt.plot(random_exmple.x, random_exmple.y, linewidth=1)
#plt.scatter(random_exmple.x[-1], random_exmple.y[-1], c='blue',s=1)
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)
plt.figure(dpi=141,figsize=(10,6))
plt.show()
#m = input("want again? (y/n)")
#if m == "n":
# break