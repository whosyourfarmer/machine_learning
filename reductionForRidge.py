import numpy as np
from sklearn import preprocessing, linear_model
import mlfunc as func
'''Training set V5 is for dimension reduction'''
train_x = np.genfromtxt('./training/train_x_V5.csv',delimiter=',')
train_y = np.genfromtxt('./training/train_y_V5.csv',delimiter=',')
#test_x = np.genfromtxt('./testing/test_x_total.csv',delimiter=',')
#test_y = np.genfromtxt('./testing/test_y_total.csv',delimiter=',')
#test_x = np.genfromtxt('./training/train_x_V1.csv',delimiter=',')
#test_y = np.genfromtxt('./training/train_y_V2.csv',delimiter=',')

scaler = preprocessing.StandardScaler().fit(train_x)
train_x_std = scaler.transform(train_x)
ridgereg = linear_model.Ridge(alpha=1e5)
ridgereg.fit(train_x_std,train_y)

for x in range(4):
	test_x = np.genfromtxt('./training/train_x_V'+str(1+x)+'.csv',delimiter=',')
	#train_new = np.zeros((train_x.shape[0],1))
	test_new = np.zeros((test_x.shape[0],1))
	for i in range(len(ridgereg.coef_)):
	 	if abs(ridgereg.coef_[i]) >= 1e-2:
	 		#train_new = np.column_stack((train_new,train_x[:,i]))
	 		test_new = np.column_stack((test_new,test_x[:,i]))
	#np.delete(train_new,0,1)
	np.delete(test_new,0,1)
	print(test_new.shape)
	np.savetxt('./training/train_x_V'+str(1+x)+'_Reduce.csv',test_new,delimiter=",")
	#np.savetxt("./testing/test_x_total_Reduce.csv",test_new,delimiter=",")