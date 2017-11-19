'''
==============================================
random forest regression with cross validation
==============================================

This code implements standardization and cross validation
at first. Then use preprocessed data and RF regression
to predict comments that will be received. 
Accuracy measure can be chosen from mae and mse.
'''
print(__doc__)

import numpy as np
from sklearn import preprocessing, ensemble
import random as rd
from sklearn.utils import shuffle
import mlfunc as func
#import math
def RFRegression(X_val,Y_val,X_tra,Y_tra,percent,figure=False):
	reg = ensemble.RandomForestRegressor(n_estimators=10,max_features=percent)
	reg.fit(X_tra,Y_tra)
	result = reg.predict(X_tra)
	maximum = 0
	for x in range(len(result)):
		maximum = max(maximum,result[x])
	err_train = func.accuracyMeasure(Y_tra,result,0.1,'prec',maximum)
	result = reg.predict(X_val)
	err_valid = func.accuracyMeasure(Y_val,result,0.1,'prec',maximum)
	#plot a figure to make a comparison
	if figure is True:
		#func.pltdiffFig(Y_val,result,'absolute')
		func.pltCurvesFig(Y_val,result)
		print('max_features',percent)
	return [err_valid,err_train]

train_x = np.genfromtxt('./training/train_x_V1.csv',delimiter=',')
train_y = np.genfromtxt('./training/train_y_V1.csv',delimiter=',')
test_x = np.genfromtxt('./testing/test_x_total.csv',delimiter=',')
test_y = np.genfromtxt('./testing/test_y_total.csv',delimiter=',')
fold, repeat = 5,1
minErr,l,maxscore = 10000,0,0
error_valid,error_train = 0,0
for lamb in [0.1,0.3,0.5,0.7,0.9]: # set a loop to choose best lambda for different versions
	error = [0,0]
	for times in range(repeat):
		train_x_rdm,train_y_rdm = shuffle(train_x,train_y,random_state = rd.randint(1,10))
		error_valid,error_train = 0,0
		for i in range(fold):
			# cross validation
			[cross_valid_x,cross_valid_y,cross_train_x,cross_train_y] = func.matrixSplit(train_x_rdm,train_y_rdm,fold,i)
			# standardize datasets regression
			scaler = preprocessing.StandardScaler().fit(cross_train_x)
			std_train_x = scaler.transform(cross_train_x)
			std_valid_x = scaler.transform(cross_valid_x)
			ret = RFRegression(std_valid_x,cross_valid_y,std_train_x,cross_train_y,lamb)
			error_valid += ret[0]
			error_train += ret[1]
		error[0] += error_valid / fold
		error[1] += error_train / fold
	for k in error:
		k /= repeat
	[error_valid,error_train] = error
	error_valid,error_train = 1/error_valid,1/error_train
	if error_valid < minErr:		# find the smallest error and its corresponding lamb
		minErr,l = error_valid,lamb
print(l)
lamb = l
# standardize datasets regression
scaler = preprocessing.StandardScaler().fit(train_x)
std_train_x = scaler.transform(train_x)
std_test_x = scaler.transform(test_x)
[test_err,train_err] = RFRegression(std_test_x,test_y,std_train_x,train_y,lamb,True)
print(test_err,train_err)