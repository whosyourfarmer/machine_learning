import CommonCluster
import K_means
import GMMCluster
import numpy as np
from collections import Counter

train_data = CommonCluster.readFile('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Training/Features_Variant_1.csv')
test_data = CommonCluster.readFile('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Testing/Features_TestSet_2.csv')

tr_x = train_data[:, :-1]
te_x = test_data[:, :-1]
#data = np.vstack((tr_x,te_x))

train_y = train_data[:, -1:]
train_y = np.ravel(train_y)
test_y = test_data[:, -1:]
test_y = np.ravel(test_y)

numOfClusters = 10
numOfComponents = 10

#np.set_printoptions(threshold=np.nan)
# K_means
#label = K_means.Kmeans(tr_x,128,numOfClusters)
#CommonCluster.findCluster(label,numOfClusters)


# GMM
label = GMMCluster.gmm_cluster(tr_x,127,numOfComponents)
CommonCluster.findCluster(label,numOfComponents)