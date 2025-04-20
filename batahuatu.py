import matplotlib.pyplot as plt
import numpy as np



x = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016])
y1 = np.array([14.31,14.43,14.69,14.41,12.19,6.83,5])
y2 = np.array([14.18,14.29,14.58,14.25,12.04,6.78,5])



plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
