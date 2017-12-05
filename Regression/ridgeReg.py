'''
======================================
ridge regression without preprocessing
======================================

This code is a template for ridge regression without
preprocessing. It can be modified to implement preprocessing
method other than standardization. Include RidgeCV()
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
reg = linear_model.RidgeCV(alphas=[1e3,1e4,1e5,1e6,1e7],cv=fold)
reg.fit(train_x,train_y)
print(reg.alpha_)
result = reg.predict(train_x)
err_train = func.accuracyMeasure(train_y,result,0.8,'mae')
result = reg.predict(test_x)
eMat = [abs(i) for i in (result - test_y)]
err_test = func.accuracyMeasure(test_y,result,0.8,'mae')
print(err_test,err_train)

#plot a figure to compare prediction results and test_y
func.pltCurvesFig(test_y,result)