'''
================================
regression w.r.t l1 norm (Lasso)
================================

This code is a template for regression with l1 norm. 
It can be modified to implement preprocessing method
other than standardization. Include LassoCV().
'''
print(__doc__)

import numpy as np
from sklearn import preprocessing, linear_model
import mlfunc as func


train_x = np.genfromtxt('./training/train_x_V3.csv',delimiter=',')
train_y = np.genfromtxt('./training/train_y_V3.csv',delimiter=',')
test_x = np.genfromtxt('./testing/test_x_1.csv',delimiter=',')
test_y = np.genfromtxt('./testing/test_y_1.csv',delimiter=',')

fold = 5
reg = linear_model.LassoCV(alphas=[1e-2,1e-1,1,1e1,1e2,1e3],cv=fold,max_iter=1000)
reg.fit(train_x,train_y)
print(reg.alpha_)
result = reg.predict(train_x)
err_train = func.accuracyMeasure(train_y,result,0.8,'mae')
result = reg.predict(test_x)
err_test = func.accuracyMeasure(test_y,result,0.8,'mae')
print(err_test,err_train)

#plot a figure to compare prediction results and test_y
func.pltCurvesFig(test_y,result)