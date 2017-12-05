'''
Common functions need to be used in the project
i.e.: input, plot of results, printing results, error measure
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def readFile(path):

    file = np.genfromtxt(path,delimiter=',')
    feature_4 = np.zeros((len(file),106))
    row = 0
    for item in file[:,3]:
        feature_4[row,int(item)-1] = 1
        row += 1
    data = np.delete(file, 3, 1)
    data = np.column_stack((feature_4,data))
    return (data)

def readFeature(path,index):

    file = np.genfromtxt(path,delimiter=',')
    feature = file[:,index-1]
    return (feature)

def findCluster(label,numOfClusters,Y,category):
    #indices = np.zeros((numOfClust ers,))
    for C in range(numOfClusters):
        indices = [i for i, x in enumerate(list(label)) if x == C]
        #x = [i for i in range(len(indices))]
        y_label = [Y[index] for index in indices]

        print(C,indices)
        print(C,y_label)
        cat_label = [category[index] for index in indices]
        print(C,cat_label)
        #print(len(indices))
        #plt.figure()
        #plt.scatter(x,y_label)
        #plt.show()

def plotCluster(label,train_y,CC1,CC5,numOfClu):

    data = []
    for C in range(numOfClu):
        indices = [i for i, x in enumerate(list(label)) if x == C]
        y = [train_y[index] for index in indices]
        c1 = [CC1[index] for index in indices]
        c2 = [CC5[index] for index in indices]
        g = (c1,c2,y)
        data.append(g)

    colors = ("red","green","blue","black","yellow","brown","white")
    groups = ("cluster1", "cluster2", "cluster3","cluster4","cluster5","cluster6", "cluster7")

    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for data, color, group in zip(data, colors, groups):
        (x, y, z) = data
        ax.scatter(x, y, z, alpha=0.5,edgecolors=color, s=30, label=group)

    ax.set_xlabel('CC1')
    ax.set_ylabel('CC5')
    ax.set_zlabel('Comment Volume')
    #plt.title('7 clusters using GMM')
    plt.legend(loc=2)
    plt.show()
