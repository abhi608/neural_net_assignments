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

w1 = np.random.uniform(-1,1,(5,10))
w2 = np.random.uniform(-1,1,(10,6))
w3 = np.random.uniform(-1,1,(6,2))

tmp_w1 = np.random.uniform(-1,1,(5,10))
tmp_w2 = np.random.uniform(-1,1,(10,6))
tmp_w3 = np.random.uniform(-1,1,(6,2))

myu = 0.01
eps = 0.01
mom = 0

for epoch in range(1000):
	error = 0
	for i in range(data.shape[0]):
		inp = data[i]
		inp = inp.reshape(5,1)
		out = output[i] # 2x1
		hidden1 = np.matmul(np.transpose(w1), inp) # 10x1
		hidden1 = sigmoid(hidden1) # 10x1
		hidden2 = np.matmul(np.transpose(w2), hidden1) # 6x1
		hidden2 = sigmoid(hidden2) # 6x1
		pred = np.matmul(np.transpose(w3), hidden2) # 2x1
		pred = sigmoid(pred) # 2x1
		nom_y = np.linalg.norm(pred-out) * np.linalg.norm(pred-out)
		err = np.linalg.norm(pred-out)
		error += err
		# ------------------------Layer-3---------------------------------------
		delta30 = (out[0] - pred[0]) * (pred[0]) * (1 - pred[0])
		delta31 = (out[1] - pred[1]) * (pred[1]) * (1 - pred[1])
		for j in range(6):
			tmp_w3[j][0] = delta30 * hidden2[j]
			tmp_w3[j][1] = delta31 * hidden2[j]
		nom_jpy3 = np.linalg.norm(tmp_w3) * np.linalg.norm(tmp_w3)
		w3 += ((myu * nom_y)/(nom_jpy3 + eps)) * tmp_w3
		# ------------------------Layer-3---------------------------------------

		# ------------------------Layer-2---------------------------------------
		delta2 = np.random.uniform(-1,1,(6,1))
		for k in range(6):
			tmp = (delta30 * w3[k][0]) + (delta31 * w3[k][1])
			delta2[k][0] = hidden2[k] * (1 - hidden2[k]) * tmp
			for j in range(10):
				tmp_w2[j][k] = delta2[k][0] * hidden1[j][0]
		nom_jpy2 = np.linalg.norm(tmp_w2) * np.linalg.norm(tmp_w2)
		w2 += ((myu * nom_y)/(nom_jpy2 + eps)) * tmp_w2
		# ------------------------Layer-2---------------------------------------

		# ------------------------Layer-1---------------------------------------
		delta1 = np.random.uniform(-1,1,(10,1))
		for k in range(10):
			tmp = 0.0
			for j in range(6):
				tmp += (delta2[j][0] * w2[k][j])
			delta1[k][0] = hidden1[k] * (1 - hidden1[k]) * tmp
			for j in range(5):
				tmp_w1[j][k] = delta1[k][0] * inp[j][0]
		nom_jpy1 = np.linalg.norm(tmp_w1) * np.linalg.norm(tmp_w1)
		w1 += ((myu * nom_y)/(nom_jpy1 + eps)) * tmp_w1
		# ------------------------Layer-1---------------------------------------

	print "Epoch: " + str(epoch) + " | Error: " + str(error/1000)





