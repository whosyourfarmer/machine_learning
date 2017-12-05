import numpy as np
import mlfunc as mf
from sklearn import linear_model


train_x = np.genfromtxt('./training/train_x_V1.csv',delimiter=',')
train_y = np.genfromtxt('./training/train_y_V1_Class.csv',delimiter=',')
test_x = np.genfromtxt('./testing/test_x_total.csv',delimiter=',')
test_y = np.genfromtxt('./testing/test_y_total_Class.csv',delimiter=',')

logreg = linear_model.LogisticRegression(penalty='l1',C=1)
logreg.fit(train_x,train_y)
result = logreg.predict(test_x)
score = mf.accuracyMeasure(test_y,result,option='accuscore')
print(score)