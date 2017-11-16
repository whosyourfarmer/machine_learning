
def LASSO_Regressor(numOfRun,numOfFold,LAMBDA,train_Data,test_Data):

    import numpy as np
    from sklearn import preprocessing as pre
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_absolute_error
    import random
    from sklearn.linear_model import LinearRegression

    tr_x = train_Data[:,:-1]
    tr_y = train_Data[:,-1:]

    te_x = test_Data[:,:-1]
    te_y = test_Data[:,-1:]

    k = numOfFold   # number of fold
    n = numOfRun   # number of iteration
    [numOfSamples,numOfFeature] = np.shape(tr_x)
    subNum = int(numOfSamples / k)
    train_x = np.zeros((numOfSamples - subNum,numOfFeature))
    train_y = np.zeros((numOfSamples - subNum,1))

    mean_train_err = np.zeros((n,len(LAMBDA)))
    mean_val_err = np.zeros((n,len(LAMBDA)))
    cv_train_err = []
    cv_val_err = []

    for i in range(n):
        index = [m for m in range(numOfSamples)]
        random.shuffle(index)
        tr_x = tr_x[index]
        tr_y = tr_y[index]

        for h in range(len(LAMBDA)):
            train_err = []
            val_err = []
            for j in range(0,k):

                train_x[0:subNum * j,:] = tr_x[0:subNum * j,:]        # training dataset
                train_x[subNum * j:numOfSamples - subNum, :] = tr_x[subNum + subNum * j:numOfSamples,:]

                train_y[0:subNum * j,:] = tr_y[0:subNum * j,:]
                train_y[subNum * j:numOfSamples - subNum, :] = tr_y[subNum + subNum * j:numOfSamples,:]

                val_x = tr_x[0 + subNum * j:subNum + subNum * j,:]     # validation dataset
                val_y = tr_y[0 + subNum * j:subNum + subNum * j,:]

                scaler = pre.StandardScaler().fit(train_x)
                train_x_std = scaler.transform(train_x)
                val_x_std = scaler.transform(val_x)

                lasso_reg = Lasso(LAMBDA[h],fit_intercept = True)
                #lasso_reg = LinearRegression();
                lasso_reg.fit(train_x_std,train_y)

                pred_train_y = lasso_reg.predict(train_x_std)
                pred_val_y = lasso_reg.predict(val_x_std)

                train_err.append(mean_absolute_error(train_y,pred_train_y))
                val_err.append(mean_absolute_error(val_y,pred_val_y))

            mean_train_err[i,h] = np.mean(train_err)
            mean_val_err[i,h] = np.mean(val_err)

    for i in range(len(LAMBDA)):
        cv_train_err.append(np.mean(mean_train_err[:,i]))
        cv_val_err.append(np.mean(mean_val_err[:,i]))

    best_alpha = LAMBDA[cv_val_err.index(min(cv_val_err))]

    scaler = pre.StandardScaler().fit(tr_x)
    tr_x_std = scaler.transform(tr_x)
    te_x_std = scaler.transform(te_x)

    lasso_reg = Lasso(best_alpha, fit_intercept=True)
    lasso_reg.fit(tr_x_std, tr_y)

    pred_train_y = lasso_reg.predict(tr_x_std)
    pred_test_y = lasso_reg.predict(te_x_std)

    lasso_err = []
    lasso_err.append(mean_absolute_error(tr_y, pred_train_y))
    lasso_err.append(mean_absolute_error(te_y, pred_test_y))


    return (lasso_err,best_alpha,cv_val_err,te_y,pred_test_y)


