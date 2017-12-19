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
from matplotlib import pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 1)

# Put your funtions here
# ...
np.random.seed(0)
if __name__ == "__main__":
	cnf=np.zeros((2,2)); a=0; st=0
	for k in range(5):
		''' choose either unbalanced either balanced and uncomment another'''
		b=make_unbalanced_dataset(3000)
		#b=make_balanced_dataset(3000)
		
		Xtr=np.array(b[0][0:1000,:])
		ytr=b[1][0:1000]
		Xte=np.array(b[0][1000:,:])
		yte=b[1][1000:]
		c=GaussianNB()
		t=GaussianNB.fit(c,Xtr,ytr)
		if k==0:
			plot_boundary(fname="Naive_Bias_Depth_%s.png" %(k),fitted_estimator=t,X=Xte,y=yte)
		pr=t.predict(Xte)
		cnf += confusion_matrix(yte,pr)
		a += round(accuracy_score(yte,pr),2)
		st += round(np.std(pr-yte),2)
	print("Accuracy:  True negative %s  False negative %s  True positive %s  False positive %s  Accuracy score %s St dev %s" %(cnf[0,0]/5,cnf[1,0]/5,cnf[1,1]/5,cnf[0,1]/5,a/5,st/5))
	pass
