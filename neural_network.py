from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy
from ipdb import set_trace


def sigmoid(z):
	z = np.array(z)
	return 1. / (1 + np.exp(-z))


def sigmoid_gradient(z):
	return(sigmoid(z) * (1 - sigmoid(z)))


def cross_entropy_cost(pred, y):
	"""
	pred[i]: array of prediction probs of size n_classes
	y[i]: one hot encoded ground truth of size n_classes
	"""
	assert len(pred) == len(y), "dimensions must match"
	cost = 0
	for i in range(len(y)):
		cost += -np.dot(y[i], np.log(pred[i]).T) - np.dot((1 - y[i]), np.log(1 - pred[i]).T)
	return cost / len(y)


class NeuralNetwork(object):
	def __init__(self):
		data = scipy.io.loadmat('db/ex3weights.mat')
		# 25, 401
		self.theta1 = data['Theta1']
		# 10, 26
		self.theta2 = data['Theta2']

	def train(self, X, y):
		pass

	def predict(self, X):
		X = np.array(X)
		n_samples = len(X)
		# set_trace()
		X_b = np.column_stack((np.ones(n_samples), X))
		z1 = np.dot(X_b, self.theta1.T)
		a1 = sigmoid(z1)
		a1 = np.column_stack((np.ones(len(z1)), a1))
		z2 = np.dot(a1, self.theta2.T)
		a2 = sigmoid(z2)
		return a2

	def cost(self, X, y):
		return cross_entropy_cost(self.predict(X), y)
