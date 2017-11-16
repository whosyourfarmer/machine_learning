'''
=========================================
basic machine learning tools or functions
=========================================

This code implements a group of basic machine learning functions. 
'''
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def accuracyMeasure(testing,prediction,percent=0,option='MAE'):
	test_s,result_s = zip(*sorted(zip(testing,prediction)))
	if option == 'MAE' or option == 'mae':
		return metrics.mean_absolute_error(test_s[int(percent*len(test_s)):],result_s[int(percent*len(result_s)):])
	elif option == 'MSE' or option == 'mse':
		return metrics.mean_squared_error(test_s[int(percent*len(test_s)):],result_s[int(percent*len(result_s)):])
	else:
		test_index = [x for x in range(len(testing))]
		predict_map = [0 for x in range(len(prediction))]
		sort_test,sort_index = zip(*sorted(zip(testing,test_index)))
		for i in range(int(percent*len(testing))):
			predict_map[sort_index[len(testing)-1-i]] = 1
		sort_predict,sort_map = zip(*sorted(zip(prediction,predict_map)))
		sum = 0
		for i in range(int(percent*len(prediction))):
			sum += sort_map[len(prediction)-1-i]
		if option == 'prec':
			return sum/int(percent*len(prediction))
		elif option == 'recall':
	 		return sum/(len(prediction)-2*int(percent*len(prediction)+2*sum))
		return 0


def matrixSplit(X_tra,Y_tra,fold,i):
	num = X_tra.shape[0]
	blockSize = int(num/fold)
	cross_Xvalid,cross_Yvalid = X_tra[i*blockSize:(i+1)*blockSize,:],Y_tra[i*blockSize:(i+1)*blockSize]
	cross_Xtrain = np.delete(X_tra,np.s_[i*blockSize:(i+1)*blockSize],0)
	cross_Ytrain = np.delete(Y_tra,np.s_[i*blockSize:(i+1)*blockSize],0)
	return [cross_Xvalid,cross_Yvalid,cross_Xtrain,cross_Ytrain]

def pltdiffFig(origin,predict,option='basic'):
		result_plt = [x for x in predict]
		test_plt = [x for x in origin]
		sort_test,sort_result = zip(*sorted(zip(test_plt,result_plt)))
		axis_x = [x for x in range(len(result_plt))]
		if option == 'absolute':
			diff = [abs(sort_result[x]-sort_test[x]) for x in range(len(sort_result))]
		else:
			diff = [sort_result[x]-sort_test[x] for x in range(len(sort_result))]
		plt.scatter(axis_x,diff,color = (1,0,0),marker = 'o',alpha=0.1)
		plt.show()
		return

def pltCurvesFig(origin,predict):
		result_plt = [x for x in predict]
		test_plt = [x for x in origin]
		sort_test,sort_result = zip(*sorted(zip(test_plt,result_plt)))
		axis_x = [x for x in range(len(result_plt))]
		plt.scatter(axis_x,sort_result,color = (0,1,0),marker = 'x')
		plt.scatter(axis_x,sort_test,color = (1,0,0),marker = 'o',alpha=1)
		plt.show()
		return