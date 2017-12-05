'''
Lin Huang EE660 Project: Facebook Comment Volume Predicition
Classification
4 types of classifiers and compare
'''

import CommonClf
import numpy as np
import LogL2_Clf
import RF_Clf
import Ada_Clf

train_data = CommonClf.readFile('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Training/Features_Variant_1.csv')
test_data = CommonClf.readFile('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Testing/Features_TestSet_2.csv')

tr_x = train_data[:, :-1]
tr_y = train_data[:, -1:]

te_x = test_data[:, :-1]
te_y = test_data[:, -1:]

[train_label,test_label] = CommonClf.Label(tr_y,te_y,60)

# L2_Logistic Regression
(LogL2_acc,best_lambda,val_acc,train_acc,te_y,pred_test_y) = LogL2_Clf.LogisticL2_Clf(5,5,[1000000],tr_x,train_label,te_x,test_label,'recall')
CommonClf.printResults(LogL2_acc,best_lambda,val_acc,train_acc)
CommonClf.plot_output(te_y,pred_test_y)

# random forest
#(rfc_acc,best_para,val_acc,train_acc,te_y,pred_test_y) = RF_Clf.RandomForest_Clf(5,5,100,[70],tr_x,train_label,te_x,test_label,'recall')
#CommonClf.printResults(rfc_acc,best_para,val_acc,train_acc)
#CommonClf.plot_output(te_y,pred_test_y)









# AdaBoost
#(ada_acc,best_lambda,val_acc,te_y,pred_test_y) = Ada_Clf.AdaBoost_Clf(5,5,[5,10,20,30],tr_x,train_label,te_x,test_label,'recall')
#CommonClf.printResults(ada_acc,best_numOfEst,val_acc)
#CommonClf.plot_output(te_y,pred_test_y)

