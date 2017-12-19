"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as linear
from sklearn import svm as svm


# Settings
np.random.seed(0)	# Fixing the "random" values
n_iter = 20       	# Iteration number to compute the metrics
test_n = 200      	# Size of the testing set


# Parameters to tune
learn_n = 1000        	# Size of the learning set
cmplx = 1		# Model complexity parameter
st_dev_noise = 1        # Standard deviation of the noise



# One linear and one non-linear regression of my choice
linear = linear.Ridge(alpha=cmplx)							# ridge regression linear model
rbf = svm.SVR(gamma=cmplx) 									# support vector regression (RBF) model, C is just one parameter to control complexity, gamma controls it too
func = [("Ridge regression", linear), ("Support Vector Regression (RBF) model", rbf)]	# and let us put them in the list


def generate_y(x):
	x = x.ravel()
	y = np.sin(x) / np.exp(-x)
	return  y


def smpl_gen(n_smpl, st_dev_noise, n_iter):
	X = np.sort(np.random.uniform(-4,4,n_smpl)) 
	if n_iter == 1:
		y = generate_y(X) 
	else:
		y = np.zeros((n_smpl, n_iter))
		for i in range(n_iter):
			y[:, i] = generate_y(X) + np.random.normal(0,st_dev_noise,n_smpl)
	X = X.reshape((n_smpl, 1))
	return X, y


def decomposition(X_test,y_test,y_predict,n_iter):
	""" The experimental protocol: decomposition on squared bias & variance & noise of the mean squared error """
	y_error = np.zeros(test_n)

	for i in range(n_iter):
		for j in range(n_iter):
			y_error += (y_test[:, j] - y_predict[:, i]) ** 2

	# computation of the mean squared error
	y_error /= (n_iter * n_iter)
	# computation of noise
	y_noise = np.var(y_test, axis=1)
	# computation of bias
	y_bias = (generate_y(X_test) - np.mean(y_predict, axis=1)) ** 2
	# computation of variance
	y_var = np.var(y_predict, axis=1)

	print('%s: %s (error) = %s (bias^2)  + %s (var) + %s (noise)' %(type_func,round(np.mean(y_error),2),round(np.mean(y_bias),2),round(np.mean(y_var),2),round(np.mean(y_noise),2)))
	return y_error,y_bias,y_var,y_noise


plt.figure(figsize=(14, 7))
bbox_args = dict(boxstyle="square",fc='white')

# Generation of learning and testing datasets
X_learn,y_learn = [],[]

for i in range(n_iter):
	X, y = smpl_gen(learn_n, st_dev_noise, 1)
	X_learn.append(X); y_learn.append(y)

X_test, y_test = smpl_gen(test_n, st_dev_noise, n_iter)



# Loop over both function and plot them
for n, (type_func, body) in enumerate(func):

	# Prediction computation
	y_predict = np.zeros((test_n, n_iter))

	for i in range(n_iter):
		body.fit(X_learn[i], y_learn[i])
		y_predict[:, i] = body.predict(X_test)

	# Invocation the requested experimental protocol adopted for the particular function
	err,eps,bias,var = decomposition(X_test,y_test,y_predict,n_iter)

	# Plot figures
	plt.subplot(2, 2, n+1)
	plt.plot(X_learn[0], y_learn[0], ".b", label=r'${y}=\frac{\sin {x}}{e^{-{x}}} + \epsilon$')#r'${y}=\frac{$\mathrm{sin}({x})}{e^{-{x}}} +noise$')

	for i in range(n_iter):
		if i == 0:
			plt.plot(X_test, y_predict[:, i], "r", alpha=0.15, label="$\^y(x)$")
		else:
			plt.plot(X_test, y_predict[:, i], "r", alpha=0.15)

	plt.plot(X_test, np.mean(y_predict, axis=1), "g",linewidth = 2,label=r"$\frac{1}{n\_iter}\sum_{n=1}^{n\_iter}\^y(x)$")
	plt.xlim([-5, 5])
	plt.ylim([-35, 10])
	plt.title(type_func)
	if n == 2 - 1:
		plt.legend(loc=(1.1, .3))

	plt.subplot(2, 2, n+3)
	plt.plot(X_test, err, "r", label="Error")
	plt.plot(X_test, eps, "b", label="Squared bias"),
	plt.plot(X_test, bias, "g", label="Variance"),
	plt.plot(X_test, var, "c", label="Noise")
	plt.yscale('log')

	plt.xlim([-5, 5])
	plt.ylim([0, 40])

	if n == 2 - 1:
		plt.legend(loc=(1.1, .5))
plt.subplots_adjust(right=.75)
plt.show()
plt.savefig("Metrics" )

