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
with open('../data.csv', 'r') as csvfile:
	all_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	count = 0
	prev_v1 = -1
	prev_v2 = -1
	for row in all_data:
		if count > 1:
			data.append([float(prev_v1), float(prev_v2), float(row[4]), float(row[6]), float(row[8])])
			output.append([float(row[0]), float(row[2])])
		count += 1
		prev_v1 = row[0]
		prev_v2 = row[2]


#-----------------------------------------------------------------------------------

data = np.asarray(data)
output = np.asarray(output)

# print data.shape, output.shape

w1 = np.random.uniform(-1,1,(5,15))
w2 = np.random.uniform(-1,1,(15,2))

prev_w1 = np.random.uniform(-1,1,(5,15))
prev_w2 = np.random.uniform(-1,1,(15,2))

tmp_w1 = np.random.uniform(-1,1,(5,15))
tmp_w2 = np.random.uniform(-1,1,(15,2))

lr = 0.01
alpha = 0.5
count = 0

for epoch in range(1000):
	error = 0
	for i in range(data.shape[0]):
		inp = data[i]
		inp = inp.reshape(5,1)
		out = output[i] # 2x1
		hidden = np.matmul(np.transpose(w1), inp) # 15x1
		hidden = sigmoid(hidden) # 15x1
		pred = np.matmul(np.transpose(w2), hidden) # 2x1
		pred = sigmoid(pred) # 2x1
		err = 0.5 * np.linalg.norm(pred-out) * np.linalg.norm(pred-out)
		error += err
		# ------------------------Layer-2---------------------------------------
		delta20 = (out[0] - pred[0]) * (pred[0]) * (1 - pred[0])
		delta21 = (out[1] - pred[1]) * (pred[1]) * (1 - pred[1])
		for j in range(15):
			tmp_w2[j][0] = w2[j][0]
			tmp_w2[j][1] = w2[j][1]
			if count > 0:
				w2[j][0] += lr * delta20 * hidden[j] + alpha * (w2[j][0] - prev_w2[j][0])
				w2[j][1] += lr * delta21 * hidden[j] + alpha * (w2[j][1] - prev_w2[j][1])
				prev_w2[j][0] = tmp_w2[j][0]
				prev_w2[j][1] = tmp_w2[j][1]
			else:
				w2[j][0] += lr * delta20 * hidden[j]
				w2[j][1] += lr * delta21 * hidden[j]
		# ------------------------Layer-2---------------------------------------

		# ------------------------Layer-1---------------------------------------
		for k in range(15):
			tmp = (delta20 * w2[k][0]) + (delta21 * w2[k][1])
			delta1 = hidden[k] * (1 - hidden[k]) * tmp
			for j in range(5):
				tmp_w1[j][k] = w1[j][k]
				if count > 0:
					w1[j][k] += lr * delta1 * inp[j][0] + alpha * (w1[j][k] - prev_w1[j][k])
					prev_w1[j][k] = tmp_w1[j][k]
				else:
					w1[j][k] += lr * delta1 * inp[j][0]
		# ------------------------Layer-1---------------------------------------
	count += 1
	print "Epoch: " + str(epoch) + " | Error: " + str(error/1000)









