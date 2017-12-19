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
from matplotlib import pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score



# (Question 2)

# Put your funtions here
# ...
d=[1,5,50,100,500]

if __name__ == "__main__":
	print("Nearest Neighbour: Standard calculation")
	for i in range(len(d)):
		cnf=np.zeros((2,2)); a=0; st=0
		for k in range(5):
			b=make_unbalanced_dataset(3000)
			Xtr=np.array(b[0][0:1000,:])
			ytr=b[1][0:1000]
			Xte=np.array(b[0][1000:,:])
			yte=b[1][1000:]
			c=KNeighborsClassifier(n_neighbors=d[i])
			t=KNeighborsClassifier.fit(c,Xtr,ytr)
			plot_boundary(fname="K_nearest_neighbors_%s.png" %(str(d[i])),fitted_estimator=t,X=Xte,y=yte)
			pr=t.predict(Xte)
			cnf += confusion_matrix(yte,pr)
			a += round(accuracy_score(yte,pr),3)
			st += round(np.std(pr-yte),2)
		print("Average accuracy if  N-value %s :  True negative %s  False negative %s  True positive %s  False positive %s  Accuracy score %s St dev %s" %(d[i],cnf[0,0]/5.,cnf[1,0]/5.,cnf[1,1]/5.,cnf[0,1]/5.,a/5., st/5.))
	print("Ten-fold cross validation")
	for i in range(len(d)):
		cnf=np.zeros((2,2)); a=0; st=0
		for k in range(5):
			b=make_unbalanced_dataset(3000)
			Xtr=np.array(b[0][0:1000,:])
			ytr=b[1][0:1000]
			Xte=np.array(b[0][1000:,:])
			yte=b[1][1000:]
			prr=np.zeros((2000))
			for k in range(10):
				if k==0:
					Xtrt,ytrt=Xtr[0:900],ytr[0:900]
				elif k==9:
					Xtrt,ytrt=Xtr[100:1000],ytr[100:1000]
				else:
					Xtrt,ytrt=np.concatenate((Xtr[0:k*100],Xtr[k*100+100:1000]),axis=0),np.concatenate((ytr[0:k*100],ytr[k*100+100:1000]),axis=0)
				c=KNeighborsClassifier(n_neighbors=d[i])
				t=KNeighborsClassifier.fit(c,Xtrt,ytrt)
				prr += t.predict(Xte)
			pr10=np.rint(prr/10.)
			a += round(accuracy_score(yte,pr10),3)
			cnf += confusion_matrix(yte,pr10)
			st += round(np.std(pr10-yte),2)
		print("Average accuracy for Ten-fold cross-validations if N-value %s :  True negative %s  False negative %s  True positive %s  False positive %s  Accuracy score %s St dev %s" %(d[i],cnf[0,0]/5.,cnf[1,0]/5.,cnf[1,1]/5.,cnf[0,1]/5.,a/5.,st/5.))
	pass
