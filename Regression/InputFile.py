#print(__doc__)
'''
Lin Huang EE660 Project: Facebook Comment Volume Predicition
Input Dataset
This code is for reading the training data and testing data
One hot encoding is used for transfer 4th categorical feature
into 106 features with value 1 or 0.

'''

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

'''
    data = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            temp = []
            for item in row:
                temp.append(float(item))
            feature_4 = [0] * 107
            feature_4[int(temp[3]) - 1] = 1
            del temp[3]
            data_row = feature_4 + temp
            data.append(data_row)

    data = np.matrix(data)
    print(data)
    return (data)
'''