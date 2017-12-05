from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score

def Kmeans(x,max_f,numOfClu_vector):

    print(np.shape(x))
    if (127<=max_f<=len(x)):
        pca = PCA(n_components=max_f)
        x = pca.fit_transform(x)

    elif (max_f<127):
        print("error")
        return

    calinski_harabaz = []
    largest_ch = -1*np.infty
    for numOfClu in numOfClu_vector:
        KM_cluster = KMeans(n_clusters=numOfClu)
        KM_cluster.fit(x)
        label = KM_cluster.labels_
        calinski_harabaz.append(calinski_harabaz_score(x,label))
        if calinski_harabaz[-1] > largest_ch:
            largest_ch = calinski_harabaz[-1]
            best_KM = KM_cluster
    label = best_KM.labels_

    best_index = calinski_harabaz.index(largest_ch)
    print(best_KM.get_params(deep=True))
    return(label,calinski_harabaz,numOfClu_vector[best_index])
