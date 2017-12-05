import numpy as np
from sklearn import preprocessing, cluster
import mlfunc as func

train_x = np.genfromtxt('./training/train_x_V1.csv',delimiter=',')
train_y = np.genfromtxt('./training/train_y_V1_Class.csv',delimiter=',')
ones_x = np.zeros(1,train_x.shape[1])
zeros_x = np.zeros(1,train_x.shape[1])
for x in range(len(train_y)):
	if train_y[x] == 1:
		ones_x = np.append(ones_x,train_x[x,:])
	else:
		zeros_x = np.append(zeros_x,train_x[x,:])
ones_x = np.delete(ones_x,0,0)
zeros_x = np.delete(zeros_x,0,0)

clusters = 2
kmean = cluster.KMeans(n_clusters=clusters).fit(mydata)
percent = [0 for x in range(clusters)]
nums = [0 for x in range(clusters)]
for x in range(len(kmean.labels_)):
	nums[kmean.labels_[x]] += 1
	if train_y[x] == 1:
		percent[kmean.labels_[x]] += 1
for x in range(clusters):
	percent[x] /= nums[x]
print(percent,nums)