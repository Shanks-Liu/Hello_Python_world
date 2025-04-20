import math 
import numpy as np

results = np.empty(8)
results2 = np.empty(8)
results3 = np.empty(8)
print(">>>>>>>>>>>>>", repr(results.shape))

for i in range(results.shape[0]):
	results[i] = np.exp(np.power((1/(i+1)), 2 * (results.shape[0] - (i+1))) * (-1 / math.e))
	results2[i] = np.exp(((results2.shape[0]-(i+1)) ** 2) * (-1 / math.e))
	results3[i] = np.exp(1 - ((i + 1)** (((i + 1) - results3.shape[0]) *2)))

print(results)	
print(results2)
print(results3)