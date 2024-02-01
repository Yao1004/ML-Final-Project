from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd

train_load = np.genfromtxt('train.csv', delimiter=',')
train_data = np.delete(train_load, 0, 0)
cols = len(train_data[0])
train_label = train_data[:, 0]
train_data = train_data[:, [x for x in range(1, cols)]]

test_load = np.genfromtxt('test.csv', delimiter=',')
test_data = np.delete(test_load, 0, 0)

# train_data , test_data , train_label , test_label = train_test_split(x,y,test_size=0.2)

adaboost = AdaBoostClassifier(n_estimators = 500)
adaboost.fit(train_data,train_label)
ty = adaboost.predict(test_data)

output = {
    'id'            : [i+17170 for i in range (len(test_data))],
    'Danceability'  : [[0]*len(test_data)]
}

output['Danceability'] = ty

pd.DataFrame(output).to_csv('out.csv', index=False)

'''mae = 0
for i in range(len(ty)) :
    mae += abs(ty[i]-test_label[i])
mae /= len(ty)

print('Eval',mae)

mae = 0

ty = adaboost.predict(train_data)
for i in range(len(ty)) :
    mae += abs(ty[i]-train_label[i])
mae /= len(ty)

print('Ein',mae)'''
