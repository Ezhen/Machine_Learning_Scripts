import numpy as np; from sklearn import metrics; from collections import Counter


########################################################################################
##########################          SVC - RBF            ###############################
########################################################################################
def rbff(X_train,y_train,X_test,C=10,gamma=0.01):
	from sklearn.svm import SVC
	knn = SVC(kernel='rbf', C=C, degree=gamma, class_weight='balanced')
	knn.fit(X_train,y_train) 
	y_predict = knn.predict(X_test)
	return y_predict


########################################################################################
##################          Random forest classifier            ########################
########################################################################################
def trees(X_train_clean,y_train,X_test_clean,nn=5):
    	from sklearn.ensemble import RandomForestClassifier
    	knn = RandomForestClassifier(n_estimators=nn, criterion='gini',class_weight='balanced')
    	knn.fit(X_train_clean,y_train) 
    	y_predict = knn.predict(X_test_clean)
	return y_predict


########################################################################################
###################           K-neigbour Classifier             ########################
########################################################################################
def kn(X_train,y_train,X_test,nn=5):
	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, n_neighbors=nn, p=2,weights='uniform')
	knn.fit(X_train,y_train) 
	y_predict = knn.predict(X_test)
	return y_predict


########################################################################################
##########################          RBM-pipe            ################################
########################################################################################
def br(X_train,y_train,X_test,ln=0.0001,ni=30,C=1,gamma=0.1):
	from sklearn.neural_network import BernoulliRBM
	from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
	linear = SVC(kernel='linear', C=C,degree=gamma,class_weight='balanced',probability=True)
	rbm = BernoulliRBM(n_components = 50,learning_rate = ln,n_iter = ni, verbose = True)
	classifier = Pipeline(steps=[('rbm', rbm), ('linear', linear)])
	classifier.fit(X_train, y_train)
	y_predict = classifier.predict(X_test)
	return y_predict


########################################################################################
#########################          AnaBoost            #################################
########################################################################################
def boost(X_train_clean,y_train,X_test_clean,maxd=5,ne=10,lr=0.01):
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=maxd),n_estimators=ne,learning_rate=lr,algorithm="SAMME")	
	bdt_discrete.fit(X_train_clean, y_train)	
	y_predict = bdt_discrete.predict(X_test_clean)
	return y_predict


########################################################################################
#########################          Bagging            ##################################
########################################################################################
def bagg(X_train_clean,y_train,X_test_clean):
	from sklearn.ensemble import BaggingClassifier
	rr = BaggingClassifier(base_estimator=None, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=True, n_jobs=1)
	rr.fit(X_train_clean, y_train)
	y_predict = rr.predict(X_test_clean)
	return y_predict


########################################################################################
#########################          SoftVote            #################################
########################################################################################
def softvote(X_train_clean,y_train,X_test_clean,w1,w2):
	from sklearn.ensemble import VotingClassifier
	from sklearn.ensemble import BaggingClassifier	
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	bg = BaggingClassifier(DecisionTreeClassifier(max_depth=3,class_weight='balanced'), n_estimators=15)
	bs = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,class_weight='balanced'),n_estimators=15,learning_rate=5,algorithm="SAMME")
	eclf = VotingClassifier(estimators=[('bagging', bg), ('boosting', bs)], voting='soft', weights=[w1,w2])
	eclf.fit(X_train_clean, y_train)
	y_predict = eclf.predict(X_test_clean)
	return y_predict


########################################################################################
############################          MLP            ###################################
########################################################################################
def mp(X_train,y_train,X_test,hid=(100,100)):
	from sklearn.neural_network import MLPClassifier
	knn = MLPClassifier(solver='adam', hidden_layer_sizes=hid)
	knn.fit(X_train,y_train) 
	y_predict = knn.predict(X_test)
	return y_predict


########################################################################################
########################          MLP + 10 Fold           ##############################
########################################################################################
def neur10(X_train,y_train,X_test):
	from sklearn.neural_network import MLPClassifier   
	clf =  MLPClassifier(solver='adam', hidden_layer_sizes=(100,100))
	y_pred=np.zeros((len(X_test),10))
	y_predict2=np.zeros((len(X_test)))
	for k in range(10):
		if k==0:
			Xtrt,ytrt=X_train[0:13249],y_train[0:13249]
		elif k==9:
			Xtrt,ytrt=X_train[1472:14720],y_train[1472:14720]
		else:
			Xtrt,ytrt=np.concatenate((X_train[0:k*1472],X_train[k*1472+1472:14720]),axis=0),np.concatenate((y_train[0:k*1472],y_train[k*1472+1472:14720]),axis=0)
		t = clf.fit(Xtrt,ytrt)  
		y_pred[:,k] = t.predict(X_test); print k
	for i in range(len(y_predict2)):
		print len(np.where(y_pred[i,:]==-1)[0])
		if len(np.where(y_pred[i,:]==-1)[0])==4:
			y_predict2[i] = -1
		else:
			y_predict2[i] = Counter(y_pred[i,:]).most_common(1)[0][0]
	return y_predict2


def pr(y_predict):
	tt = []
    	for i in range(10):
		tt.append(len(np.where(y_predict==i)[0]))
   	tt.append(len(np.where(y_predict==-1)[0])); print tt


def prediction_test(y_predict):
	def ex(n1,n2,v):
		return len(np.where(np.array(y_predict[n1:n2])==v)[0])
	def ln(n1,n2):
		return float(len(np.arange(n1,n2)))
	s_1 = round(100 * (ex(0,1000,-1) + ex(6000,7000,-1) + ex(13500,14000,-1) + ex(15750,16250,-1) + ex(20000,20750,-1)) / (ln(0,1000) + ln(6000,7000) + ln(8750,10000) + ln(13500,14000) + ln(15750,16250) + ln(20000,20750)),3)
	s0 = round(100 * ex(21000,22567,0) / ln(21000,22567),3)
	s1 = round(100 * ex(10250,11750,1) / ln(10250,11750),3)
	s2 = round(100 * ex(18000,19500,2) / ln(18000,19500),3)
	s3 = round(100 * ex(16500,17750,3) / ln(16500,17750),3)
	s4 = round(100 * ex(4500,5750,4) / ln(4500,5750),3)
	s5 = round(100 * ex(2750,4250,5) / ln(2750,4250),3)
	s6 = round(100 * ex(14250,15500,6) / ln(14250,15500),3)
	s7 = round(100 * ex(12000,13250,7) / ln(12000,13250),3)
	s8 = round(100 * ex(1250,2500,8) / ln(1250,2500),3)
	s9 = round(100 * ex(7250,8500,9) / ln(7250,8500),3)
	s_digit = round(100 * ( ex(21000,22567,0) + ex(10250,11750,1) + ex(18000,19500,2) + ex(16500,17750,3) + ex(4500,5750,4) + ex(2750,4250,5) + ex(14250,15500,6) + ex(12000,13250,7) + ex(1250,2500,8) + ex(7250,8500,9) ) / (ln(21000,22567) + ln(10250,11750) + ln(18000,19500) + ln(16500,17750) + ln(4500,5750) + ln(2750,4250) + ln(14250,15500) + ln(12000,13250) + ln(1250,2500) + ln(7250,8500) ),3)
	total = round(100 * (ex(0,1000,-1) + ex(6000,7000,-1) + ex(13500,14000,-1) + ex(15750,16250,-1) + ex(20000,20750,-1) + ex(21000,22567,0) + ex(10250,11750,1) + ex(18000,19500,2) + ex(16500,17750,3) + ex(4500,5750,4) + ex(2750,4250,5) + ex(14250,15500,6) + ex(12000,13250,7) + ex(1250,2500,8) + ex(7250,8500,9) ) / (ln(0,1000) + ln(6000,7000) + ln(8750,10000) + ln(13500,14000) + ln(15750,16250) + ln(20000,20750) + ln(21000,22567) + ln(10250,11750) + ln(18000,19500) + ln(16500,17750) + ln(4500,5750) + ln(2750,4250) + ln(14250,15500) + ln(12000,13250) + ln(1250,2500) + ln(7250,8500) ),3)
	print "  ACCURACY TEST [%] (covers 82.275% of the testing dataset)"
	print "-1:", s_1, ", 0:", s0, ", 1:", s1, ", 2:", s2, ", 3:", s3, ", 4:", s4, ", 5:", s5, ", 6:", s6, ", 7:", s7, ", 8:", s8, ", 9:", s9, ", digit class:", s_digit, ", total:", total
	return

def number_class(y_predict):
    	for i in range(10):
		print i, len(np.where(y_predict==i)[0])
   	print -1, len(np.where(y_predict==-1)[0])

