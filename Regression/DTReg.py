'''
Lin Huang EE660 Project: Facebook Comment Volume Predicition
Regression:
Using CART algorithm to predict the comment volume
Cross Validation is used to do model selection with respect to
different depth
MAE, Hits, 2 types of error measure are utilized

Parameter:
numOfRun: number of iterations
numOfFold: number of folds for Cross Validation
depth: a list of depth we use to select the best model
train_Data, test_Data: training/testing Dataset
error_measure: choose a specific error measure
ratio: ratio of the data set
'''

import numpy as np
from sklearn import preprocessing as pre
from sklearn.tree import DecisionTreeRegressor
import random
import CommonReg

def DT_Regressor(numOfRun,numOfFold,depth,train_Data,test_Data,err_measure,ratio):

    tr_x = train_Data[:,:-1]
    tr_y = train_Data[:,-1:]
    tr_y = np.ravel(tr_y)

    te_x = test_Data[:,:-1]
    te_y = test_Data[:,-1:]
    te_y = np.ravel(te_y)

    k = numOfFold   # number of fold
    n = numOfRun   # number of iteration
    [numOfSamples,numOfFeature] = np.shape(tr_x)
    subNum = int(numOfSamples / k)
    train_x = np.zeros((numOfSamples - subNum,numOfFeature))
    train_y = [0]*(numOfSamples - subNum)
    train_y = np.array(train_y)

    mean_train_err = np.zeros((n,len(depth)))
    mean_val_err = np.zeros((n,len(depth)))
    cv_train_err = []
    cv_val_err = []

    for i in range(n):
        index = [m for m in range(numOfSamples)]
        random.shuffle(index)
        tr_x = tr_x[index]
        tr_y = tr_y[index]

        for h in range(len(depth)):
            train_err = []
            val_err = []
            for j in range(0,k):

                train_x[0:subNum * j,:] = tr_x[0:subNum * j,:]        # training dataset
                train_x[subNum * j:numOfSamples - subNum, :] = tr_x[subNum + subNum * j:numOfSamples,:]

                train_y[0:subNum * j] = tr_y[0:subNum * j]
                train_y[subNum * j:numOfSamples - subNum] = tr_y[subNum + subNum * j:numOfSamples]

                val_x = tr_x[0 + subNum * j:subNum + subNum * j,:]     # validation dataset
                val_y = tr_y[0 + subNum * j:subNum + subNum * j]

                scaler = pre.StandardScaler().fit(train_x)   # standardization
                train_x_std = scaler.transform(train_x)
                val_x_std = scaler.transform(val_x)

                dt_reg = DecisionTreeRegressor(max_depth = depth[h])
                dt_reg.fit(train_x_std,train_y)

                pred_train_y = dt_reg.predict(train_x_std)
                pred_val_y = dt_reg.predict(val_x_std)

                train_err.append(CommonReg.errorMeasure(train_y, pred_train_y,err_measure,ratio))
                val_err.append(CommonReg.errorMeasure(val_y, pred_val_y,err_measure,ratio))

            mean_train_err[i,h] = np.mean(train_err)
            mean_val_err[i,h] = np.mean(val_err)

    for i in range(len(depth)):
        cv_train_err.append(np.mean(mean_train_err[:,i]))
        cv_val_err.append(np.mean(mean_val_err[:,i]))

    best_depth = depth[cv_val_err.index(min(cv_val_err))]  # model selection

    scaler = pre.StandardScaler().fit(tr_x)   # standardization
    tr_x_std = scaler.transform(tr_x)
    te_x_std = scaler.transform(te_x)

    dt_reg = DecisionTreeRegressor(max_depth = best_depth)
    dt_reg.fit(tr_x_std, tr_y)

    pred_train_y = dt_reg.predict(tr_x_std)
    pred_test_y = dt_reg.predict(te_x_std)

    dt_err = []
    dt_err.append(CommonReg.errorMeasure(tr_y, pred_train_y,err_measure,ratio))
    dt_err.append(CommonReg.errorMeasure(te_y, pred_test_y,err_measure,ratio))


    return (dt_err,best_depth,cv_val_err,cv_train_err,te_y,pred_test_y)


