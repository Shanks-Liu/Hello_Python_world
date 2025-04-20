import numpy as np

data = np.array([13.3,6.2,15.5,16.3,13.6,7.3,5,6.4])
#print(data.shape[0])
#print(data.size)
#beta = int(input("give a number beta:  "))


beta = np.array([10,9,8,7,6,5,4,3,2,1,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])

results = np.empty(data.shape[0])

for j in range(beta.shape[0]):
	for i in range(data.shape[0]):
		results[i] = np.power(np.average(np.power(data[i:].copy(), beta[j])), 1/beta[j])
		
	print(np.round(results, 2))
#print(data)