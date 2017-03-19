#!/usr/bin/env python
import pandas as pd
import numpy as np
from IPython import embed
import ipdb


def least_square_error(estimate, y):
	"""
	Compute Least Square Error.

	Parameters
	----------
	estimate : numpy array of shape [n_samples]
		Prediction data
	y : numpy array of shape [n_samples]
		Target values

	Returns
	-------
	cost : total cost over all samples
	"""
	estimate = np.array(estimate)
	y = np.array(y)
	assert(len(estimate)==len(y)), "dimensions don't match"
	sum = 0
	for i in range(len(y)):
		sum += (estimate[i]-y[i])**2
	return sum / (2 * len(y))


class LinearRegression(object):

	def __init__(self):
		# theta : model parameter of size 2
		# J : cost history
		self.theta = np.array([0, 0])
		# self.theta = np.random.randn(2)
		self.J = []

	def fit(self, X, y, alpha = 0.01, n_iter = 1000):
		"""
		Fit linear model.

		Parameters
		----------
		X : numpy array of shape [n_samples,n_features]
			Training data
		y : numpy array of shape [n_samples, n_targets]
			Target values

		Returns
		-------
		self : returns an instance of self.
		"""
		assert(len(X)==len(y)),"dimensions don't match"
		for i in range(n_iter):
			pred = self.predict(X)
			# print "pred & y ", pred[:5], y[:5]
			# gradient descent
			self.theta -= float(alpha)/len(y) * np.dot((pred - y), X)
			# print "theta: ", self.theta
			self.J.append(least_square_error(pred,y))
			# print "J: ", self.J

	def predict(self, X):
		"""
		Compute predictions (vectorized)

		Parameters
		----------
		X : numpy array of shape [n_samples, n_features]

		Returns
		-------
		pred : numpy array of shape [n_samples]
		"""

		# X: [[1,x1],..., [1,xn]], theta: [t1;t2]
		dim = len(X)
		ones = np.ones(dim)
		X = np.stack((ones, np.array(X)), axis = 1)
		return np.dot(X, self.theta.T)

	def save(self, X):
		"""
		Save parameters of the model.
		Parameters
		----------
		Returns
		-------
		"""

def main():
	data = pd.read_csv('db/ex1data1.txt', names=['population', 'profit'])
	X = data.population.values
	y = data.profit.values
	model = LinearRegression()
	model.fit(X,y)
	pred = model.predict(X)




if __name__ == '__main__':
	main()
