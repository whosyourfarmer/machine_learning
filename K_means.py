from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans

def Kmeans(x,max_f,numOfClusters):

    print(np.shape(x))
    if (127<=max_f<=len(x)):
        pca = PCA(n_components=max_f)
        x = pca.fit_transform(x)

    elif (max_f<127):
        print("error")
        return

    print(np.shape(x))
    KM_cluster = KMeans(n_clusters=numOfClusters)
    KM_cluster.fit(x)
    label = KM_cluster.labels_
    return(label)