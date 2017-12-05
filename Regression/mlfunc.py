'''
=========================================
basic machine learning tools or functions
=========================================

This code implements a group of basic machine learning functions. 
'''
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def accuracyMeasure(testing,prediction,percent=0,option='MAE',maximum=1):
	test_s,result_s = zip(*sorted(zip(testing,prediction),key=lambda num: num[0]))
	if option == 'MAE' or option == 'mae':
		return metrics.mean_absolute_error(test_s[int(percent*len(test_s)):],result_s[int(percent*len(result_s)):])
	elif option == 'MSE' or option == 'mse':
		return metrics.mean_squared_error(test_s[int(percent*len(test_s)):],result_s[int(percent*len(result_s)):])
	elif option == 'accuscore':
		return metrics.accuracy_score(testing,prediction)
	elif option == 'precscore':
		return metrics.precision_score(testing,prediction,average='binary')
	elif option == 'recallscore':
		return metrics.recall_score(testing,prediction)
	elif option == 'f1score':
		return metrics.f1_score(testing,prediction)
	elif option == 'aucscore':
		return metrics.roc_auc_score(testing,prediction)
	else:
		predict_index = [x for x in range(len(prediction))]
		predict_map = [0 for x in range(len(prediction))]
		sort_predict,sort_index = zip(*sorted(zip(prediction,predict_index),key=lambda num: num[0]))
		for i in range(int(percent*len(prediction))):
			predict_map[sort_index[len(prediction)-1-i]] = 1
		sort_test,sort_map = zip(*sorted(zip(testing,predict_map),key=lambda num: num[0]))
		sum = 0
		for i in range(int(percent*len(testing))):
			sum += sort_map[len(testing)-1-i]
		if option == 'hits':
			return sum/int(percent*len(testing))
		elif option == 'recall':
	 		return sum/int(percent*len(prediction))
		elif option == 'auc':
			test_index = [x for x in range(len(testing))]
			test_map = [0 for x in range(len(testing))]
			sort_test,sort_index = zip(*sorted(zip(testing,test_index),key=lambda num: num[0]))
			for i in range(int(percent*len(testing))):
				test_map[sort_index[len(testing)-1-i]] = 1
			score_predict = [0 for x in range(len(prediction))]
			for x in range(len(prediction)):
				if maximum < prediction[x]:
					score_predict[x] = 1
				elif prediction[x] < 0:
					score_predict[x] = 0
				else:
					score_predict[x] = prediction[x]/maximum
			return metrics.roc_auc_score(test_map,score_predict)
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
		sort_test,sort_result = zip(*sorted(zip(test_plt,result_plt),key=lambda num: num[0]))
		axis_x = [x for x in range(len(result_plt))]
		if option == 'absolute':
			diff = [abs(sort_result[x]-sort_test[x]) for x in range(len(sort_result))]
		else:
			diff = [sort_result[x]-sort_test[x] for x in range(len(sort_result))]
		plt.figure()
		plt.scatter(axis_x,diff,color = (1,0,0),marker = 'o',alpha=0.1)
		plt.show()
		return

def pltCurvesFig(origin,predict):
		result_plt = [x for x in predict]
		test_plt = [x for x in origin]
		sort_test,sort_result = zip(*sorted(zip(test_plt,result_plt),key=lambda num: num[0]))
		axis_x = [x for x in range(len(result_plt))]
		plt.figure()
		plt.scatter(axis_x,sort_result,color = (1,1,0),marker = 'x')
		plt.scatter(axis_x,sort_test,color = (0,0,1),marker = 'o',alpha=1)
		plt.show()
		return