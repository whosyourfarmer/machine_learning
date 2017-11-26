'''
Lin Huang EE660 Project: Facebook Comment Volume Predicition
Classification:
Using AdaBoost based on CART to predict the comment volume
L2 regularization is used
Cross Validation is used to do model selection with respect to
different number of estimators
Confusion matrix is plotted and AUC/recall/precision is used to measure the accuracy

Parameter:
numOfRun: number of iterations
numOfFold: number of folds for Cross Validation
numOfEst: a list of number of estimators
train_Data, test_Data: training/testing Dataset
acc_measure: choose a specific accuracy measure
'''

import numpy as np
from sklearn import preprocessing as pre
from sklearn.ensemble import AdaBoostClassifier
import random
import CommonClf

def AdaBoost_Clf(numOfRun,numOfFold,numOfEst,train_x,train_label,test_x,test_label,acc_measure):

    tr_x = train_x
    tr_y = train_label

    te_x = test_x
    te_y = test_label

    k = numOfFold   # number of fold
    n = numOfRun   # number of iteration
    [numOfSamples,numOfFeature] = np.shape(tr_x)
    subNum = int(numOfSamples / k) # divide training set into k subsets
    train_x = np.zeros((numOfSamples - subNum,numOfFeature))
    train_y = [0]*(numOfSamples - subNum)
    train_y = np.array(train_y)


    mean_train_acc = np.zeros((n,len(numOfEst)))
    mean_val_acc = np.zeros((n,len(numOfEst)))
    cv_train_acc = []
    cv_val_acc = []

    for i in range(n):
        index = [m for m in range(numOfSamples)]
        random.shuffle(index)  # randomize training set at each iteration
        tr_x = tr_x[index]
        tr_y = tr_y[index]

        for h in range(len(numOfEst)):
            train_acc = []
            val_acc = []
            for j in range(0,k):

                train_x[0:subNum * j,:] = tr_x[0:subNum * j,:]        # training dataset
                train_x[subNum * j:numOfSamples - subNum, :] = tr_x[subNum + subNum * j:numOfSamples,:]

                train_y[0:subNum * j] = tr_y[0:subNum * j]
                train_y[subNum * j:numOfSamples - subNum] = tr_y[subNum + subNum * j:numOfSamples]

                val_x = tr_x[0 + subNum * j:subNum + subNum * j,:]     # validation dataset
                val_y = tr_y[0 + subNum * j:subNum + subNum * j]

                scaler = pre.StandardScaler().fit(train_x)     # standardization
                train_x_std = scaler.transform(train_x)
                val_x_std = scaler.transform(val_x)

                adaclf = AdaBoostClassifier(n_estimators=numOfEst[h],algorithm='SAMME.R')
                adaclf.fit(train_x_std,train_y)

                pred_train_y = adaclf.predict(train_x_std)
                pred_val_y = adaclf.predict(val_x_std)

                train_acc.append(CommonClf.clf_accuracy(train_y, pred_train_y, acc_measure))
                val_acc.append(CommonClf.clf_accuracy(val_y, pred_val_y, acc_measure))

            mean_train_acc[i,h] = np.mean(train_acc)
            mean_val_acc[i,h] = np.mean(val_acc)

    for i in range(len(numOfEst)):
        cv_train_acc.append(np.mean(mean_train_acc[:,i]))
        cv_val_acc.append(np.mean(mean_val_acc[:,i]))

    best_numOfEst = numOfEst[cv_val_acc.index(max(cv_val_acc))]   # model selection

    scaler = pre.StandardScaler().fit(tr_x)    # standardization
    tr_x_std = scaler.transform(tr_x)
    te_x_std = scaler.transform(te_x)

    adaclf = AdaBoostClassifier(n_estimators=numOfEst[h], algorithm='SAMME.R')
    adaclf.fit(tr_x_std, tr_y)

    pred_train_y = adaclf.predict(tr_x_std)
    pred_test_y = adaclf.predict(te_x_std)

    ada_acc = []
    ada_acc.append(CommonClf.clf_accuracy(tr_y, pred_train_y,'cnf'))
    ada_acc.append(CommonClf.clf_accuracy(te_y, pred_test_y,'cnf'))

    # AUC
    #tr_y_score = adaclf.predict_proba(tr_x_std)
    te_y_score = adaclf.predict_proba(te_x_std)
    #CommonClf.AUC_Measure(tr_y,tr_y_score)
    CommonClf.AUC_Measure(te_y,te_y_score)


    return (ada_acc,best_numOfEst,cv_val_acc,te_y,pred_test_y)


