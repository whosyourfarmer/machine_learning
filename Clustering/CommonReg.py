'''
Common functions need to be used in the project
i.e.: input, plot of results, printing results, error measure
'''

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

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

def plot_output(true_y,pred_y):

    index = [i for i in range(len(true_y))]
    (true, pred) = zip(*sorted(zip(true_y, pred_y),key=lambda x:x[0]))

    plt.figure()
    plt.plot(index, pred)
    plt.plot(index, true)
    plt.show()

def printResults(error, best_para, val_err):
    print("Validation error of all models: ", val_err)
    print("Best para: ", best_para)
    print("Training error: ", error[0])
    print("Testing error: ", error[1])

def errorMeasure(true_y,pred_y,err_measure='MAE',ratio=1):
    (pred_sort,true) = zip(*sorted(zip(pred_y,true_y),key=lambda x:x[0]))
    size = int(ratio * len(true_y))
    # MAE
    if (err_measure == 'MAE'):
        return metrics.mean_absolute_error(pred_sort[-size:],true[-size:])
    # Hits
    if (err_measure == 'Hits'):
        if (np.all(pred_y == [0] * len(pred_y))):
            return 1
        point = [0] * len(true_y)
        point[-size:] = [1] * size
        (true_sort,index) = zip(*sorted(zip(true,point),key=lambda x:x[0]))
        total = sum(index[-size:])
        return 1 - total / size    # return the probability of not being on the top 10%
