'''
Lin Huang EE660 Project: Facebook Comment Volume Predicition
Regression:
Call 4 types of regressors and compare
'''


import SVR
import DTReg
import CommonReg
import AdaBoost_Reg
import LASSO

train_data = CommonReg.readFile('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Training/Features_Variant_1.csv')
test_data = CommonReg.readFile('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Testing/Features_TestSet_2.csv')

# lasso
#[lasso_err,best_alpha,val_err,train_err,te_y,pred_test_y] = LASSO.LASSO_Regressor(5,5,[1000],train_data,test_data,'Hits',0.1)
#lasso_err = [1-p for p in lasso_err]
#val_err = [1-p for p in val_err]
#train_err = [1-p for p in train_err]
#CommonReg.printResults(lasso_err,best_alpha,val_err,train_err)
#CommonReg.plot_output(te_y,pred_test_y)

#[lasso_err,best_alpha,val_err,train_err,te_y,pred_test_y] = LASSO.LASSO_Regressor(5,5,[1000],train_data,test_data,'MAE',1)
#CommonReg.printResults(lasso_err,best_alpha,val_err,train_err)
#CommonReg.plot_output(te_y,pred_test_y)

# cart
#[dt_err,best_depth,val_err,train_err,te_y,pred_test_y] = DTReg.DT_Regressor(5,5,[9],train_data,test_data,'Hits',0.1)
#dt_err = [1-p for p in dt_err]
#val_err = [1-p for p in val_err]
#train_err = [1-p for p in train_err]
#CommonReg.printResults(dt_err,best_depth,val_err,train_err)
#CommonReg.plot_output(te_y,pred_test_y)

#[dt_err,best_depth,val_err,train_err,te_y,pred_test_y] = DTReg.DT_Regressor(5,5,[9],train_data,test_data,'MAE',1)
#CommonReg.printResults(dt_err,best_depth,val_err,train_err)
#CommonReg.plot_output(te_y,pred_test_y)

# Adaboost
#[ada_err,best_numOfEst,val_err,train_err,te_y,pred_test_y] = AdaBoost_Reg.AdaBoost_Regressor(5,5,[30],train_data,test_data,'Hits',0.1)
#ada_err = [1-p for p in ada_err]
#val_err = [1-p for p in val_err]
#train_err = [1-p for p in train_err]
#CommonReg.printResults(ada_err,best_numOfEst,val_err,train_err)
#CommonReg.plot_output(te_y,pred_test_y)

[ada_err,best_numOfEst,val_err,train_err,te_y,pred_test_y] = AdaBoost_Reg.AdaBoost_Regressor(5,5,[30],train_data,test_data,'MAE',1)
CommonReg.printResults(ada_err,best_numOfEst,val_err,train_err)
CommonReg.plot_output(te_y,pred_test_y)

# svr
#[svr_err,best_C,val_err,te_y,pred_test_y] = SVR.SVR_Regressor(1,2,[10],train_data,test_data,'Hits',0.1)
#svr_err = [1-p for p in svr_err]
#val_err = [1-p for p in val_err]
#CommonReg.printResults(svr_err,best_C,val_err)
#CommonReg.plot_output(te_y,pred_test_y)

#[svr_err,best_C,val_err,te_y,pred_test_y] = SVR.SVR_Regressor(1,2,[10],train_data,test_data,'MAE',1)
#CommonReg.printResults(svr_err,best_C,val_err)
#CommonReg.plot_output(te_y,pred_test_y)



# randomize training/testing set in order to eliminate the effect of Hits
# lasso
#[lasso_err,best_alpha,val_err,te_y,pred_test_y] = LASSO_Rand.LASSO_Regressor(5,5,[0.00001,0.001,0.1,1,5,10,100],train_data,test_data,'Hits',0.1)
#lasso_err = [1-p for p in lasso_err]
#val_err = [1-p for p in val_err]
#print(lasso_err,best_alpha,val_err)
#Common.plot_output(te_y,pred_test_y)

# cart
#[dt_err,best_depth,val_err,te_y,pred_test_y] = DTReg_Rand.DT_Regressor(5,5,[1,2,3,4,5,6,7,8,9],train_data,test_data,'Hits',0.1)
#dt_err = [1-p for p in dt_err]
#val_err = [1-p for p in val_err]

#print(dt_err,best_depth,val_err)
#Common.plot_output(te_y,pred_test_y)