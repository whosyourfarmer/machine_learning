'''
======================================
ridge regression w.r.t standardization
======================================

This code implements standardization and cross validation
at first. Then use preprocessed data and ridge regression
to predict comments that will be received. 
Accuracy measure can be chosen from mae and mse.
'''
print(__doc__)

import numpy as np
from sklearn import preprocessing, linear_model
import random as rd
from sklearn.utils import shuffle
import mlfunc as func

def ridgeRegression(X_val,Y_val,X_tra,Y_tra,regur,figure):
	ridgereg = linear_model.Ridge(alpha=regur)
	ridgereg.fit(X_tra,Y_tra)
	result = ridgereg.predict(X_tra)
	err_train = func.accuracyMeasure(Y_tra,result,0.1,'prec')
	result = ridgereg.predict(X_val)
	err_valid = func.accuracyMeasure(Y_val,result,0.1,'prec')
	#plot a figure to make a comparison
	if figure is True:
		func.pltCurvesFig(Y_val,result)
	return [err_valid,err_train]

train_x = np.genfromtxt('./training/train_x_V1.csv',delimiter=',')
train_y = np.genfromtxt('./training/train_y_V1.csv',delimiter=',')
test_x = np.genfromtxt('./testing/test_x_total.csv',delimiter=',')
test_y = np.genfromtxt('./testing/test_y_total.csv',delimiter=',')
#print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
fold, repeat = 5,3
minErr,l = 10000,0
error_valid,error_train = 0,0
for lamb in [1e2,1e3,1e4,1e5,1e6,1e7]: # set a loop to choose best lambda for different versions
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
			ret = ridgeRegression(std_valid_x,cross_valid_y,std_train_x,cross_train_y,lamb,False)
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
[test_err,train_err] = ridgeRegression(std_test_x,test_y,std_train_x,train_y,lamb,True)
print(test_err,train_err)
