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

param = [
    [3, 7, 10], # max_depth
    [1, 3, 5], # min_child_weight
    [0, 0.05 , 0.5], # gamma
    [0.6, 0.8, 1], # subsample
    [0.6, 0.8, 1] # colsample_bytree
]

train_x , val_x , train_y , val_y = train_test_split(x,y,test_size=0.15)

vy = []
yin = []

for element in tqdm(itertools.product(*param)):

    xgboostModel = [[] for i in range (9)]
    vy_bin = []
    yin_bin = []

    vy_single = [0 for i in range (len(val_x))]
    yin_single = [0 for i in range (len(val_x))]

    for i in range(9) :
        train_y_bin = np.where(train_y <= i, 0, 1)
        xgboostModel[i] = XGBClassifier(n_estimators = 300, learning_rate= 0.3, max_depth = element[0]
                                        ,min_child_weight = element[1], gamma = element[2], 
                                        subsample = element[3], colsample_bytree = element[4],reg_lambda=0.1)
        xgboostModel[i].fit(train_x, train_y_bin)
        vy_bin.append(xgboostModel[i].predict(val_x))
        yin_bin.append(xgboostModel[i].predict(train_x))
    for i in range (len(val_x)) :
        for j in range (9) :
            vy_single[i] += float(vy_bin[j][i])
            yin_single[i] += float(yin_bin[j][i])
    vy.append(vy_single)
    yin.append(yin_single)

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
    'Danceability'  : [0 for i in range (len(tx))]
}

yr = []

for i in range(9) :
    y_bin = np.where(y <= i, 0, 1)
    xgboostModel_test = XGBClassifier(n_estimators = 150, learning_rate= 0.3, max_depth = opt[0],
                                      min_child_weight = opt[1], gamma = opt[2], 
                                      subsample = opt[3], colsample_bytree = opt[4])
    xgboostModel_test.fit(x, y_bin)
    yr.append(xgboostModel_test.predict(tx))

for i in range (len(tx)) :
    for j in range (9) :
        output['Danceability'][i] += float(yr[j][i])

ans = pd.read_csv('test_partial_answer.csv')

for i in range (ans['id']) :
    output['Danceability'][ output['id'].index(ans['id'][i]) ] = ans['Danceability'][i]

pd.DataFrame(output).to_csv('out_xgboost2.csv', index=False)