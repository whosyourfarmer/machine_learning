''' 
==========================================
preprocessing part for regression problems
==========================================

This code is for preprocessing training data and testing data 
used in regression. Transfer 4th feature (multi-categories) 
into 106 featureswith value 1 or 0. 
Split training data (testing data) into train_x, train_y 
(test_x, test_y) and then standardize train_x and test_x.
'''
print(__doc__)

import numpy as np
from sklearn import preprocessing

mydata = np.genfromtxt('./Facebook_Dataset/Training/Features_Variant_5.csv',delimiter=',')
#mydata = np.genfromtxt('./Facebook_Dataset/Testing/TestSet/Test_Case_10.csv',delimiter=',')
#mydata = np.genfromtxt('./Facebook_Dataset/Testing/Features_TestSet.csv',delimiter=',')
mydata_y = mydata[:,-1]
mydata_y = mydata[:,-1]
mydata_x = mydata[:,0:-1]
myfeature = np.zeros((mydata.shape[0],106))
row = 0
for i in mydata[:,3]:
	if int(i) < 107:
		myfeature[row,int(i)-1] = 1
	row += 1
matrix = np.column_stack((mydata_x,myfeature))
matrix = np.delete(matrix,3,1)
np.savetxt("./training/train_x_V5.csv",matrix,delimiter=",")
np.savetxt("./training/train_y_V5.csv",mydata_y,delimiter=",")
#np.savetxt("./testing/test_x_10.csv",matrix,delimiter=",")
#np.savetxt("./testing/test_y_10.csv",mydata_y,delimiter=",")