import CommonCluster
import K_means
import GMMCluster
import AggloCluster
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

category = CommonCluster.readFeature('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Training/Features_Variant_1.csv',4)
category = np.ravel(category)
CC1 = CommonCluster.readFeature('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Training/Features_Variant_1.csv',30)
CC1 = np.ravel(CC1)
CC5 = CommonCluster.readFeature('/Users/linhuang/Desktop/Lin_Work/EE660_Project/Data/Training/Features_Variant_1.csv',34)
CC5 = np.ravel(CC5)

clusters_vector = [7]

#np.set_printoptions(threshold=np.nan)

# GMM
[label,bic,best_NumOfCom] = GMMCluster.gmm_cluster(train_data,127,clusters_vector)
CommonCluster.findCluster(label,best_NumOfCom,train_y,category)
CommonCluster.plotCluster(label,train_y,CC1,CC5,7)
print(best_NumOfCom)
print(bic)

# Agglomerative
#[label,ch,best_NumOfClu] = AggloCluster.Agg_cluster(train_data,127,clusters_vector)
#CommonCluster.findCluster(label,best_NumOfClu,train_y,category)
#CommonCluster.plotCluster(label,train_y,CC1,CC5,7)
#print(best_NumOfClu)
#print(ch)

# K_means
#[label,ch,best_NumOfClu] = K_means.Kmeans(tr_x,127,clusters_vector)
#CommonCluster.findCluster(label,best_NumOfClu,train_y,category)
#print(best_NumOfClu)
#print(ch)
