from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from sklearn.linear_model import SGDClassifier

'''param = [
    [-3,-2,-1,0,1,2,3], # eta
    [1, 2, 3], # degree
]'''

param = [
    [2], # eta
    [2], # degree
]

train_load = np.genfromtxt('train_knn.csv', delimiter=',')
train_data = np.delete(train_load, 0, 0)
cols = len(train_data[0])

y = train_data[:, 0]
x = train_data[:, [i for i in range(1, cols)]]

train_x, val_x, train_y, val_y  = train_test_split(x,y,test_size=0.15)

val_load = np.genfromtxt('ans_data.csv', delimiter=',')
val_data = np.delete(val_load, 0, 0)
cols_val = len(val_data[0])
y_ans = val_data[:, -1]
x_ans = val_data[:, [i for i in range(0, cols_val-1)]]

val_x = np.concatenate([val_x , x_ans])
val_y = np.concatenate([val_y , y_ans])

vy = []

for i in param[1] :
    z_train = PolynomialFeatures(degree=i).fit_transform(train_x)
    z_val = PolynomialFeatures(degree=i).fit_transform(val_x)
    for j in tqdm(param[0]) :

        vy_bin = []
        vy_single = [0 for i in range (len(val_x))]

        for k in range(9) :

            train_y_bin = np.where(train_y <= k, 0, 1)
            logreg = SGDClassifier(eta0=10**j,max_iter=100, loss='log_loss')
            logreg.fit(z_train,train_y_bin)
            vy_bin.append(logreg.predict(z_val))

        for i in range (len(val_x)) :
            for j in range (9) :
                vy_single[i] += float(vy_bin[j][i])

        vy.append(vy_single)

mae = [0 for i in range (len(list(itertools.product(*param))))]

for i in range (len(mae)) :
    for j in range(len(val_y)) :
        mae[i] += abs(vy[i][j] - val_y[j])
    mae[i] /= len(val_y)

print(min(mae))

opt = list(itertools.product(*param))[mae.index(min(mae))]
print(opt)

test_load = np.genfromtxt('test_knn.csv', delimiter=',')
tx = np.delete(test_load, 0, 0)
output = {
    'id'            : [i+17170 for i in range (len(tx))],
    'Danceability'  : [0 for i in range(len(tx))]
}

z = PolynomialFeatures(degree=opt[1]).fit_transform(x)
z_ans = PolynomialFeatures(degree=opt[1]).fit_transform(x_ans)
tz = PolynomialFeatures(degree=opt[1]).fit_transform(tx)
yr = []

for i in range(9) :
    y_bin = np.where(y <= i, 0, 1)
    ans_y_bin = np.where(y_ans <= i, 0, 1)
    Logreg_test = SGDClassifier(eta0=10**opt[0],max_iter=100, loss='log_loss')
    Logreg_test.fit(np.concatenate([z , z_ans]), np.concatenate([y_bin , ans_y_bin]))
    yr.append(Logreg_test.predict(tz))

for i in range (len(tx)) :
    for j in range (9) :
        output['Danceability'][i] += float(yr[j][i])

ans = pd.read_csv('test_partial_answer.csv')

for i in range (len(ans['id'])) :
    output['Danceability'][ list(output['id']).index(ans['id'][i]) ] = ans['Danceability'][i]

pd.DataFrame(output).to_csv('log-reg.csv', index=False)