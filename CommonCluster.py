'''
Common functions need to be used in the project
i.e.: input, plot of results, printing results, error measure
'''

import numpy as np
def readFile(path):
    import numpy as np

    file = np.genfromtxt(path,delimiter=',')
    feature_4 = np.zeros((len(file),106))
    row = 0
    for item in file[:,3]:
        feature_4[row,int(item)-1] = 1
        row += 1
    data = np.delete(file, 3, 1)
    data = np.column_stack((feature_4,data))
    return (data)

def findCluster(label,numOfClusters):
    #indices = np.zeros((numOfClusters,))
    for C in range(numOfClusters):
        indices = [i for i, x in enumerate(list(label)) if x == C]
        print(C,indices)
        print(len(indices))