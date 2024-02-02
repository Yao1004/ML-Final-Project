from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from copy import deepcopy

train_load = np.genfromtxt('train.csv', delimiter=',')
train_data = np.delete(train_load, 0, 0)
cols = len(train_data[0])

y = train_data[:, 0]
x = train_data[:, [i for i in range(1, cols)]]

c = [-2,0,2]

train_x, val_x, train_y, val_y  = train_test_split(x,y,test_size=0.2)

tyr = [[] for i in range (9)]
ty = [[] for i in range (9)]
tin = []
for i in range (3) :
    z_train = PolynomialFeatures(degree=i+1).fit_transform(train_x)
    z_val = PolynomialFeatures(degree=i+1).fit_transform(val_x)
    for j in c :

        logreg = [[] for i in range (9)]
        for k in range(9) :

            train_y_cpy = deepcopy(train_y)
            val_y_cpy = deepcopy(val_y)
            for l in range (len(train_y_cpy)) :
                train_y_cpy[l] = 1 if train_y_cpy[l] > k else 0
            for l in range (len(val_y_cpy)) :
                val_y_cpy[l] = 1 if val_y_cpy[l] > k else 0
            
            logreg[k] = linear_model.LogisticRegression(C=10**j,max_iter=1000,solver='sag')
            logreg[k].fit(z_train,train_y_cpy)

        tyr.append(logreg.predict(z_val))
        tin.append(logreg.predict(z_train))

        for sa in range (len(z_val)) :
            for sb in range (len(tyr)) :
                ty[sa] += tyr[sb][sa]

mae = [0,0,0,0,0,0,0,0,0]
for i in range (9) :
    for j in range(len(val_y)) :
        mae[i] += abs(ty[i][j]-val_y[j])
    mae[i] /= len(ty[i])

Ein = [0,0,0,0,0,0,0,0,0]
for i in range (9) :
    for j in range(len(train_y)) :
        Ein[i] += abs(tin[i][j]-train_y[j])
    Ein[i] /= len(tin[i])

print(mae)
print(Ein)

degree_opt = (mae.index(min(mae)) // 3) + 1 
C_opt = (mae.index(min(mae)) % 3) - 1

print(mae)
print(Ein)

test_load = np.genfromtxt('test.csv', delimiter=',')
test_data = np.delete(test_load, 0, 0)

output = {
    'id'            : [i+17170 for i in range (len(test_data))],
    'Danceability'  : [[0]*len(test_data)]
}

z = PolynomialFeatures(degree=degree_opt).fit_transform(x)
logreg = [[] for i in range (9)]
for k in range(9) :

    test_y_cpy = deepcopy(train_y)
    for l in range (len(train_y_cpy)) :
        train_y_cpy[l] = 1 if train_y_cpy[l] > k else 0
    for l in range (len(val_y_cpy)) :
        val_y_cpy[l] = 1 if val_y_cpy[l] > k else 0
    
    logreg[k] = linear_model.LogisticRegression(C=10**C_opt,max_iter=2000,solver='sag')
    logreg[k].fit(z_train,train_y_cpy)

logreg.fit(z, y)
output['Danceability'] = logreg.predict(PolynomialFeatures(degree=degree_opt).fit_transform(test_data))

pd.DataFrame(output).to_csv('out_reg.csv', index=False)