from __future__ import division
import numpy as np
import random
import math

def generate(y, u):
	new_y = (y/(1+(y*y))) + (u*u*u)
	return new_y

def gauss(inp, c, sig):
	z = np.linalg.norm(inp-c)
	phi = math.exp(-(z*z)/(2*sig*sig))
	return phi

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

lr1 = 0.005
lr2 = 0.005
lr3 = 0.005
hidden = 100
num_epoch = 1500
sigma = np.ones((100,1))

c = random.sample(data, hidden)
c = np.asarray(c) # 100x2

data = np.asarray(data)
output = np.asarray(output)

w = np.random.uniform(-1,1,(hidden,1))


for epoch in range(num_epoch):
	error = 0
	for i in range(data.shape[0]):
		inp = data[i]
		out = output[i]
		phi = [] #100
		for j in range(hidden):
			phi.append(gauss(inp, c[j], sigma[j][0]))
		phi = np.asarray(phi)
		phi = phi.reshape(1, hidden)
		y = np.matmul(phi, w)
		err = 0.5 * (y[0][0] - out) * (y[0][0] - out)
		error += err
		for j in range(hidden):
			w[j][0] += lr1 * (out - y) * phi[0][j]
		for j in range(hidden):
			c[j] += lr2 * (out - y[0][0]) * w[j][0] * (phi[0][j]/(sigma[j][0] * sigma[j][0])) * (inp - c[j])
		for j in range(hidden):
			sigma[j][0] += lr3 * (out - y[0][0]) * w[j][0] * (phi[0][j]/(sigma[j][0] * sigma[j][0] * sigma[j][0])) * np.linalg.norm(inp-c[j]) * np.linalg.norm(inp-c[j])
	print "Epoch: " + str(epoch) + " | Error: " + str(error/1000)
