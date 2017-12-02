
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import calinski_harabaz_score
from sklearn.cluster import AgglomerativeClustering

def Agg_cluster(x,max_f,numOfClu_vector):

    print(np.shape(x))
    if (127<=max_f<=len(x)):
        pca = PCA(n_components=max_f)
        x = pca.fit_transform(x)

    elif (max_f<127):
        print("error")
        return

    print(np.shape(x))
    calinski_harabaz = []
    largest_ch = -1*np.infty
    for numOfClu in numOfClu_vector:
        Agg_cluster = AgglomerativeClustering(n_clusters=numOfClu)
        Agg_cluster.fit(x)
        label = Agg_cluster.labels_
        calinski_harabaz.append(calinski_harabaz_score(x,label))
        if calinski_harabaz[-1] > largest_ch:
            largest_ch = calinski_harabaz[-1]
            best_Agg = Agg_cluster
    label = best_Agg.labels_

    best_index = calinski_harabaz.index(largest_ch)
    print(best_Agg.get_params(deep=True))
    return(label,calinski_harabaz,numOfClu_vector[best_index])

