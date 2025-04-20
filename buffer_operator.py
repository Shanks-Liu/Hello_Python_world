import numpy as np


x = np.array([13.3,6.2,15.5,16.3,13.6,7.3,5,6.4])

results = np.empty(8)

beta = int(input("give the number of beta:  "))

for i in range(results.shape[0]):
	results[i] = np.power(((results.shape[0] - (i+1) + 1) * x[i] ** (2*beta)) / np.sum((x[i:] ** beta)), 1/beta)

print(results)
	