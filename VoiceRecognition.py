#! /usr/bin/env python
# -*- coding: utf-8 -*-


## Import packages
import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import time
from collections import Counter
from sklearn import linear_model, metrics
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
from collections import OrderedDict
from functions import *


def make_submission(y_predict, name=None, date=True):
    n_elements = len(y_predict)

    if name is None:
      name = 'submission'
    if date:
      name = name + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M'))

    with open('Submission/'+name + ".txt", 'w') as f:
        f.write('"ID","PREDICTION"\n')
        for i in range(n_elements):
            if np.isnan(y_predict[i,1]):
                raise ValueError('NaN detected!')
            line = '{:0.0f},{:0.0f}\n'.format(y_predict[i,0],y_predict[i,1])
            f.write(line)
    print("Submission file successfully written!")


if __name__ == "__main__":

    from sklearn.preprocessing import normalize

    # Load data_train
    X_train_rough = np.load('X_train.npy')[:,1:]
    X_test_rough = np.load('X_test.npy')[:,1:]
    y_train = np.load('y_train.npy')

    # Data preprocessing
    X_train_rough[X_train_rough < -200] = 0.0
    X_train = normalize(X_train_rough[:], axis=1)

    # Split the training array into two parts to perform further the investigation on the efficiency of different machine learning algorithms
    X_train_train,y_train_train,X_train_test,y_train_test=[],[],[],[]; m,k = 0,0
    for i in range(len(X_train)):
	if (i+1)%3==0:
		X_train_train.append([]);
		X_train_train[k]=list(X_train[i])
		y_train_train.append(y_train[i,1])
		k=k+1
	else:
		X_train_test.append([])
		X_train_test[m]=list(X_train[i])
		y_train_test.append(y_train[i,1])
		m=m+1	

    X_test_rough[X_test_rough < -200] = 0.0
    X_test = normalize(X_test_rough[:], axis=1)

    y_predict = np.zeros((len(X_test),2))
    y_predict[:,0] = np.arange(0,len(X_test)) # Ids for test sample

    # CHOOSE A CLASSIFIER

    # SVC - RBF
    #y_train_predict = rbff(X_train_train,y_train_train,X_train_test)
    #y_predict[:,1] = rbff(X_train,y_train[:,1],X_test)

    # Random forest classifier
    #y_train_predict = trees(X_train_train,y_train_train,X_train_test)
    #y_predict[:,1] = trees(X_train,y_train[:,1],X_test)

    # K-neigbour Classifier
    #y_train_predict = kn(X_train_train,y_train_train,X_train_test)
    #y_predict[:,1] = kn(X_train,y_train[:,1],X_test)

    # RBM-pipe
    #y_train_predict = br(X_train_train,y_train_train,X_train_test)
    #y_predict[:,1] = br(X_train,y_train[:,1],X_test)

    # AnaBoost
    #y_train_predict = boost(X_train_train,y_train_train,X_train_test)
    #y_predict[:,1] = boost(X_train,y_train[:,1],X_test)

    # Bagging
    #y_train_predict = bagg(X_train_train,y_train_train,X_train_test)
    #y_predict[:,1] = bagg(X_train,y_train[:,1],X_test)

    # SoftVote
    #y_train_predict = softvote(X_train_train,y_train_train,X_train_test,1,1)
    #y_predict[:,1] = softvote(X_train,y_train[:,1],X_test,1,1)

    # MLP
    #y_train_predict = mp(X_train_train,y_train_train,X_train_test)
    #y_predict[:,1] = mp(X_train,y_train[:,1],X_test)

    # Neural network + 10 Fold cross validation
    y_predict[:,1] = neur10(X_train,y_train[:,1],X_test)

    # output accuracy
    print "TRAIN METRICS [frac]"
    #print(metrics.classification_report(y_train_predict,y_train_test))
    prediction_test(y_predict[:,1])

    #y_predict[:,1] = neur10(X_train[::1],y_train[::1,1],X_test)

    #make_submission(y_predict, name="toy_submission")
