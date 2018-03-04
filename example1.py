from __future__ import division
import numpy as np
import random

def generate(y, u):
	new_y = (y/(1+(y*y))) + (u*u*u)
	return new_y

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def mod_sigmoid(x):
	return (3 / (1 + np.exp(-x))) - 1.5

data = []
output = []

#--------------------------------------------------------
y = 0.5
for i in range(1000):
	u = random.uniform(-1,1)
	tmp = generate(y, u)
	data.append([y, u])
	output.append(tmp)
	y = tmp

#--------------------------------------------------------

data = np.asarray(data)
output = np.asarray(output)

print data.shape, output.shape

w1 = np.random.uniform(-1,1,(2,15))
w2 = np.random.uniform(-1,1,(15,1))

lr = 0.005
for epoch in range(10000):
	error = 0
	for i in range(data.shape[0]):
		inp = data[i]
		inp = inp.reshape(2,1)
		out = output[i]
		hidden = np.matmul(np.transpose(w1), inp) # 50x1
		hidden = sigmoid(hidden) # 50x1
		pred = np.matmul(np.transpose(w2), hidden)
		pred = mod_sigmoid(pred)
		pred = pred[0][0]
		err = 0.5 * (pred - out) * (pred - out)
		error += err
		delta2 = (out - pred) * (pred + 1.5) * (1 - ((pred + 1.5)/3))
		w2 += lr * delta2 * hidden
		for j in range(2):
			for k in range(15):
				tmp = 0
				for l in range(1):
					tmp += delta2 * w2[k][l]
				# print tmp
				delta1 = hidden[k] * (1 - hidden[k]) * tmp
				w1[j][k] += lr * delta1 * inp[j][0]

	print "Epoch: " + str(epoch) + " | Error: " + str(error/1000)









