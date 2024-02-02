from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
import itertools

train_load = np.genfromtxt('train_knn.csv', delimiter=',')
train_data = np.delete(train_load, 0, 0)
cols = len(train_data[0])
y = train_data[:, 0]
x = train_data[:, [i for i in range(1, cols)]]

'''param = [
    [3, 7, 10], # max_depth
    [1, 3, 5], # min_child_weight
    [0, 0.05 , 0.5], # gamma
    [0.6, 0.9], # subsample
    [0.6, 0.9] # colsample_bytree
]'''

param = [
    [7, 10], # max_depth
    [3, 5], # min_child_weight
    [0, 0.05], # gamma
    [0.9], # subsample
    [0.9] # colsample_bytree
]

train_x , val_x , train_y , val_y = train_test_split(x,y,test_size=0.15)

val_load = np.genfromtxt('ans_data.csv', delimiter=',')
val_data = np.delete(val_load, 0, 0)
cols_val = len(val_data[0])
y_ans = val_data[:, 0]
x_ans = val_data[:, [i for i in range(1, cols_val)]]

val_x = np.concatenate([val_x , x_ans])
val_y = np.concatenate([val_y , y_ans])

vy = []
opt = []

for t in range(9) :

    train_y_bin = np.where(train_y <= t, 0, 1)
    vy_bin = []

    for element in tqdm(itertools.product(*param)):
        
        xgboostModel = XGBClassifier(n_estimators = 200, learning_rate= 0.3, max_depth = element[0]
                                        ,min_child_weight = element[1], gamma = element[2], 
                                        subsample = element[3], colsample_bytree = element[4],reg_lambda=0.1)
        xgboostModel.fit(train_x, train_y_bin)
        vy_bin.append(xgboostModel.predict(val_x))

    err = [0 for i in range (len(list(itertools.product(*param))))]

    for i in range (len(err)) :
        for j in range(len(val_y)) :
            err[i] += 0 if vy_bin[i][j] == val_y[j] else 1
        err[i] /= len(val_y)

    print(min(err))

    opt.append(list(itertools.product(*param))[err.index(min(err))])
    print(opt[t])

test_load = np.genfromtxt('test_knn.csv', delimiter=',')
tx = np.delete(test_load, 0, 0)
output = {
    'id'            : [i+17170 for i in range (len(tx))],
    'Danceability'  : [0 for i in range (len(tx))]
}

yr = []

for i in range(9) :
    y_bin = np.where(y <= i, 0, 1)
    ans_y_bin = np.where(y_ans <= i, 0, 1)
    xgboostModel_test = XGBClassifier(n_estimators = 150, learning_rate= 0.3, max_depth = opt[i][0],
                                      min_child_weight = opt[i][1], gamma = opt[i][2], 
                                      subsample = opt[i][3], colsample_bytree = opt[i][4])
    xgboostModel_test.fit(np.concatenate([x , x_ans]), np.concatenate([y_bin , ans_y_bin]))
    yr.append(xgboostModel_test.predict(tx))

for i in range (len(tx)) :
    for j in range (9) :
        output['Danceability'][i] += float(yr[j][i])

ans = pd.read_csv('test_partial_answer.csv')

for i in range (len(ans['id'])) :
    output['Danceability'][ list(output['id']).index(ans['id'][i]) ] = ans['Danceability'][i]

pd.DataFrame(output).to_csv('out_xgboost3.csv', index=False)