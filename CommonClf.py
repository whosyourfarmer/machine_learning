'''
Common functions need to be used in the project
i.e.: confusion matrix, labeling, plot of results, error measure

'''

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics

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

def Label(train_y,test_y,threshold = 100):
    tr_label = []
    te_label = []
    tr_y = np.ravel(train_y)
    te_y = np.ravel(test_y)

    for x in tr_y:
        if x >= threshold:
            tr_label.append(1)
        else:
            tr_label.append(0)
    for x in te_y:
        if x >= threshold:
            te_label.append(1)
        else:
            te_label.append(0)

    return (np.array(tr_label), np.array(te_label))

def plot_output(true_y,pred_y):

    index = [i for i in range(len(true_y))]
    (true, pred) = zip(*sorted(zip(true_y, pred_y),key=lambda x:x[0]))
    plt.figure()
    plt.scatter(index,pred,color = (1,0,0),marker = 'o',alpha=1)
    plt.plot(index,true)
    plt.show()

def printResults(error,best_para,val_err):
    print("Validation error/accuracy of all models: ", val_err)
    print("Best para: ", best_para)
    print("Training error/accuracy: ", error[0])
    print("Testing error/accuracy: ", error[1])

#Plot confusion matrix
#reference: http://scikit-learn.org/stable/index.html

from sklearn.model_selection import train_test_split

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def clf_accuracy(Y_true, Y_pred,acc_measure):

    # using recall as the acc_measure
    if (acc_measure == 'recall'):
        return metrics.recall_score(Y_true, Y_pred)
    # compute confusion matrix and plot the results
    if (acc_measure == 'cnf'):
        target = np.array(['Class 0','Class 1'])
        cnf_matrix = metrics.confusion_matrix(Y_true, Y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=target,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=target, normalize=True,
                              title='Normalized confusion matrix')

        #plt.show()
        return
    if (acc_measure == 'AUC'):
        return