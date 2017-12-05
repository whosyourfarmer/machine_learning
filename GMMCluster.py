
from sklearn.decomposition import PCA
import CommonCluster
import numpy as np
from sklearn.mixture import GaussianMixture

def gmm_cluster(x,max_f,numOfCom_vector):

    print(np.shape(x))
    if (127<=max_f<=len(x)):
        pca = PCA(n_components=max_f)
        x = pca.fit_transform(x)

    elif (max_f<127):
        print("error")
        return

    print(np.shape(x))
    bic = []
    lowest_bic = np.infty
    #cv_types = ['spherical', 'tied', 'diag', 'full']
    #for cv_type in cv_types:
    for numOfCom in numOfCom_vector:
        GMM_cluster = GaussianMixture(n_components=numOfCom)
        GMM_cluster.fit(x)
        bic.append(GMM_cluster.bic(x))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_GMM = GMM_cluster
    label = best_GMM.predict(x)

    best_index = bic.index(lowest_bic) % len(numOfCom_vector)
    print(best_GMM.get_params(deep=True))
    return(label,bic,numOfCom_vector[best_index])

