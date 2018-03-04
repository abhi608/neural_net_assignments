from __future__ import division
import numpy as np
import random
import csv
import ast

def generate(y, u):
	new_y = (y/(1+(y*y))) + (u*u*u)
	return new_y

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def mod_sigmoid(x):
	return (3 / (1 + np.exp(-x))) - 1.5

data = []
output = []

#----------------------------Reading test data--------------------------------------
with open('data.csv', 'r') as csvfile:
	all_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	flag = True
	min = 10000
	max = -99999
	for row in all_data:
		if flag:
			flag = False
		else:
			data.append([float(row[4]), float(row[6]), float(row[8])])
			output.append([float(row[0]), float(row[2])])
			if float(row[0]) < min:
				min = float(row[0])
			if float(row[2]) < min:
				min = float(row[2])
			if float(row[0]) > max:
				max = float(row[0])
			if float(row[2]) > max:
				max = float(row[2])

#-----------------------------------------------------------------------------------

data = np.asarray(data)
output = np.asarray(output)

print data.shape, output.shape
# print min, max

w1 = np.random.uniform(-1,1,(3,15))
w2 = np.random.uniform(-1,1,(15,2))

lr = 0.05
mom = 0
for epoch in range(1000):
	error = 0
	for i in range(data.shape[0]):
		inp = data[i]
		inp = inp.reshape(3,1)
		out = output[i] # 2x1
		hidden = np.matmul(np.transpose(w1), inp) # 15x1
		hidden = sigmoid(hidden) # 15x1
		pred = np.matmul(np.transpose(w2), hidden) # 2x1
		pred = sigmoid(pred) # 2x1
		err = np.linalg.norm(pred-out)
		error += err
		delta20 = (out[0] - pred[0]) * (pred[0]) * (1 - pred[0])
		delta21 = (out[1] - pred[1]) * (pred[1]) * (1 - pred[1])
		for j in range(15):
			w2[j][0] += (lr / (1-mom)) * delta20 * hidden[j]
			w2[j][1] += (lr / (1-mom)) * delta21 * hidden[j]

		for k in range(15):
			tmp = (delta20 * w2[k][0]) + (delta21 * w2[k][1])
			delta1 = hidden[k] * (1 - hidden[k]) * tmp
			for j in range(3):
				w1[j][k] += (lr / (1-mom)) * delta1 * inp[j][0]
		# print out[0], pred[0], out[1], pred[1]

	print "Epoch: " + str(epoch) + " | Error: " + str(error/1000)









