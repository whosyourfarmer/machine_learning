
from sklearn.decomposition import PCA
import CommonCluster
import numpy as np
from sklearn.mixture import GaussianMixture

def gmm_cluster(x,max_f,numOfCom):

    print(np.shape(x))
    if (127<=max_f<=len(x)):
        pca = PCA(n_components=max_f)
        x = pca.fit_transform(x)

    elif (max_f<127):
        print("error")
        return

    print(np.shape(x))
    GMM_cluster = GaussianMixture(n_components=numOfCom)
    GMM_cluster.fit(x)
    label = GMM_cluster.predict(x)
    return(label)

