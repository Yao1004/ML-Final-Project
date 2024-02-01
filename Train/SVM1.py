from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
import itertools

train_load = np.genfromtxt('train_knn.csv', delimiter=',')
train_data = np.delete(train_load, 0, 0)
cols = len(train_data[0])
y = train_data[:, 0]
x = train_data[:, [i for i in range(1, cols)]]

param = [
    [10], # C
    [0.01], # gamma
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
vin = []

for element in tqdm(itertools.product(*param)):

    svmModel = SVC(C=element[0], gamma=element[1])
    svmModel.fit(train_x, train_y)
    vy.append(svmModel.predict(val_x))
    vin.append(svmModel.predict(train_x))

mae = [0 for i in range (len(list(itertools.product(*param))))]

for i in range (len(mae)) :
    for j in range(len(val_y)) :
        mae[i] += abs(vy[i][j] - val_y[j])
    mae[i] /= len(val_y)

print(min(mae))

inmae = 0
for i in range (len(train_x)) :
    inmae += abs(vin[mae.index(min(mae))][i] - train_y[i])
inmae /= len(train_x)
print(inmae)

opt = list(itertools.product(*param))[mae.index(min(mae))]
print(opt)

test_load = np.genfromtxt('test_knn.csv', delimiter=',')
tx = np.delete(test_load, 0, 0)
output = {
    'id'            : [i+17170 for i in range (len(tx))],
    'Danceability'  : [0 for i in range (len(tx))]
}

svmModel_test = SVC(C=opt[0],gamma=opt[1])
svmModel_test.fit(np.concatenate([x , x_ans]), np.concatenate([y , y_ans]))
output['Danceability'] = svmModel_test.predict(tx)

ans = pd.read_csv('test_partial_answer.csv')

for i in range (len(ans['id'])) :
    output['Danceability'][ list(output['id']).index(ans['id'][i]) ] = ans['Danceability'][i]

pd.DataFrame(output).to_csv('svm1.csv', index=False)