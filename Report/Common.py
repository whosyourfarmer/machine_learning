

import numpy as np
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

def plot_output(train_y):
    train_y_sort = np.sort(train_y)

    plt.figure()
    plt.plot(train_y_sort)
    plt.ylabel(r'comment volume', {'color': 'k', 'fontsize': 15})

    #np.set_printoptions(threshold=np.nan)
    #print(train_y_sort[38000])
    #print(len(train_y_sort))
    #print(38000/len(train_y_sort))

    plt.scatter(list(train_y_sort).index(60),60, 50, color ='blue')

    plt.annotate(r'(39903,60)',
             xy=(39703,70), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=15,
             arrowprops=dict(arrowstyle="->"),horizontalalignment='right')

    plt.show()

def draw_bar(quants, measure):

    ind = np.linspace(0.5, 9.5, 6)   # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence


    if measure == 'mae':
        plt.bar(ind, quants, width)
        plt.ylabel('Mean Absoulte Error')
        plt.title('Regression Algorithms Performance(MAE)')
    elif measure == 'hits':
        plt.bar(ind, quants, width,color=[0,0,0])
        plt.ylabel('Hits(10%)')
        plt.title('Regression Algorithms Performance(Hits)')
    plt.xlabel('Regression Algorithms')
    plt.xticks(ind, ('Ridge', 'LASSO', 'OMP', 'CART', 'RandomForest', 'AdaBoost'))
    ax = plt.axes()
    ax.yaxis.grid(linestyle='--')
    plt.show()

def error_plot(tra_err,val_err):
    ind = range(7)
    plt.figure()
    #plt.plot(tra_err)
    plt.ylabel('Validation Error')
    plt.xlabel('lambda')
    plt.xticks(ind,('0.0001', '0.01', '1', '5', '10','100', '1000'))
    plt.plot(val_err)
    plt.show()












