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

train_x = np.genfromtxt('./training/train_x_V1.csv',delimiter=',')
train_y = np.genfromtxt('./training/train_y_V1.csv',delimiter=',')
test_x = np.genfromtxt('./testing/test_x_total.csv',delimiter=',')
test_y = np.genfromtxt('./testing/test_y_total.csv',delimiter=',')

fold = 5
reg = linear_model.RidgeCV(alphas=[1e3,1e4,1e5,1e6,1e7],cv=fold)
reg.fit(train_x,train_y)
print(reg.alpha_)
result = reg.predict(test_x)
eMat = [abs(i) for i in (result - test_y)]
err_test = np.dot(np.transpose(eMat),np.ones((len(eMat),1))) / test_y.shape[0]
result = reg.predict(train_x)
eMat = [abs(i) for i in (result - train_y)]
err_train = np.dot(np.transpose(eMat),np.ones((len(eMat),1))) / train_y.shape[0]
print(err_test,err_train)