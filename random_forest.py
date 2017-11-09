'''
==============================================
random forest regression w.r.t standardization
==============================================

This code is a template for random forest regression. 
It can be modified to implement other preprocessing methods. 
Include RandomForestRegressor().
'''
print(__doc__)

import numpy as np
from sklearn import preprocessing, ensemble
import mlfunc as func


train_x = np.genfromtxt('./training/train_x_V3.csv',delimiter=',')
train_y = np.genfromtxt('./training/train_y_V3.csv',delimiter=',')
test_x = np.genfromtxt('./testing/test_x_1.csv',delimiter=',')
test_y = np.genfromtxt('./testing/test_y_1.csv',delimiter=',')

scaler = preprocessing.StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

n_estimators = 10
reg = ensemble.RandomForestRegressor(n_estimators=n_estimators)
reg.fit(train_x,train_y)
result = reg.predict(train_x)
err_train = func.accuracyMeasure(train_y,result,0.8,'mae')
result = reg.predict(test_x)
err_test = func.accuracyMeasure(test_y,result,0.8,'mae')
print(err_test,err_train)

#plot a figure to compare prediction results and test_y
func.pltCurvesFig(test_y,result)