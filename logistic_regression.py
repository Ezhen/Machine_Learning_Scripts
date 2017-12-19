"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from plot import plot_boundary



class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate


    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

	theta=[0,0] # initial theta is a equal to 1
	for x in xrange(c.n_iter):
		new_theta = []
		for j in xrange(len(theta)):
			corr = c.learning_rate * Loss_Function(X,y,theta,j)
			#print ('Loss:',corr)
			theta_corr = theta[j] - corr
			new_theta.append(theta_corr)
		theta = new_theta
	self.theta=new_theta
	return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

	ypr=np.zeros((len(X)))
	for i in range(len(X)):
		ypr[i]=Sigmoid(self.theta,X[i])
	ypr=np.rint(ypr)
	return ypr

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
	a=LogisticRegressionClassifier.predict(self, X)
	b=np.zeros((len(X),2))
	for i in range(len(a)):
		if a[i]==0:
			b[i,0],b[i,1]=1,0
		else:
			b[i,0],b[i,1]=0,1
	return b


# Some useful functions
def Sigmoid(theta,x):
        """ Just sigmoid function"""
	z = x[0]*theta[0] + x[1]*theta[1]
	s = 1 / (1.0 + np.exp(-z))
	return s

def Loss_Function(X,Y,theta,j):
	Err = 0.0
	for i in xrange(len(Y)):
		xi = X[i]; xij = xi[j]
		ypr = Sigmoid(theta,X[i])
		error = (ypr - Y[i]) * xij
		Err += error
	J = Err  / len(Y)
	return J

niter=[50] #[10,200,1000]
learningrate=[0.01,0.1,1,10]
# Main body
if __name__ == "__main__":
	for i in range(len(niter)):
		for j in range(len(learningrate)):
			cnf=np.zeros((2,2)); a=0; st=0
			ni=niter[i]
			lr=learningrate[j]
			for k in range(5): # five generations of the dataset
				b=make_unbalanced_dataset(3000)
				Xtr=np.array(b[0][0:1000,:])
				ytr=b[1][0:1000]
				Xte=np.array(b[0][1000:,:])
				yte=b[1][1000:]
				c=LogisticRegressionClassifier(n_iter=ni,learning_rate=lr)
				t=LogisticRegressionClassifier.fit(c,Xtr,ytr)
				plot_boundary(fname="Logistic_regression_learn_rate_%s_n_iter_%s.png" %(lr,ni),fitted_estimator=t,X=Xte,y=yte)
				pr=t.predict(Xte)
				cnf += confusion_matrix(yte,pr)
				a += round(accuracy_score(yte,pr),3)
				st += round(np.std(pr-yte),2)
			print("Average accuracy if Learn Rate = %s & Iteration N = %s:  True negative %s  False negative %s  True positive %s  False positive %s  Accuracy score %s  St dev %s" %(lr,ni,cnf[0,0]/5.,cnf[1,0]/5.,cnf[1,1]/5.,cnf[0,1]/5.,a/5.,st/5.)); c=0
	pass
