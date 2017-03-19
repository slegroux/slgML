#!/usr/bin/env python
import pandas as pd
import numpy as np
import math
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
	assert(len(estimate) == len(y)), "dimensions don't match"
	sum = 0
	for i in range(len(y)):
		sum += (estimate[i] - y[i])**2
	return sum / (2 * len(y))


def mean(X):
	sum = 0
	for v in X:
		sum += v
	return float(sum) / len(X)


def std(X):
	sum = 0
	for v in X:
		sum += (v - mean(X))**2
	return math.sqrt(sum / (len(X) - 1))


def cov(X, Y):
	sum = 0
	for i in range(len(X)):
		sum += (X[i] - mean(X)) * (Y[i] - mean(Y))
	return sum / (len(X) - 1)


def pearson_correlation_coefficient(X, Y):
	assert(len(X) == len(Y)), "dimensions don't match"
	return float(cov(X, Y)) / (std(X) * std(Y))


class LinearRegression(object):
	def __init__(self):
		# theta : model parameter of size 2
		# J : cost history
		self.theta = np.array([0.0, 0.0])
		# self.theta = np.random.randn(2)
		self.J = []

	def fit(self, X, y, alpha=0.01, n_iter=1000):
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
		assert(len(X) == len(y)), "dimensions don't match"
		dim = len(X)
		# X_b: includes bias term
		ones = np.ones(dim)
		X_b = np.stack((ones, np.array(X)), axis=1)
		for i in range(n_iter):
			pred = self.predict(X)
			# print "pred & y ", pred[:5], y[:5]
			# gradient descent
			# ipdb.set_trace()
			self.theta -= float(alpha) / len(y) * np.dot((pred - y), X_b)
			# print "theta: ", self.theta
			self.J.append(least_square_error(pred, y))
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
		# include bias term in X
		ones = np.ones(dim)
		X_b = np.stack((ones, np.array(X)), axis=1)
		return np.dot(X_b, self.theta.T)

	def save(self, X):
		"""
		Save parameters of the model.
		Parameters
		----------
		Returns
		-------
		"""

	def score(self, X, y):
		return pearson_correlation_coefficient(self.predict(X), y)

def main():
	data = pd.read_csv('db/ex1data1.txt', names=['population', 'profit'])
	X = data.population.values
	y = data.profit.values
	model = LinearRegression()
	model.fit(X, y)
	pred = model.predict(X)


if __name__ == '__main__':
	main()
