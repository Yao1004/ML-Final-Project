from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from tqdm import tqdm

train_load = np.genfromtxt('train.csv', delimiter=',')
train_data = np.delete(train_load, 0, 0)
cols = len(train_data[0])
y = train_data[:, 0]
x = train_data[:, [x for x in range(1, cols)]]

ty = []
knn = []

for t in tqdm(range (100)) :

    train_data , test_data , train_label , test_label = train_test_split(x,y,test_size=0.2)
    knn.append(KNeighborsClassifier(n_neighbors=50, leaf_size=2, p=10))
    knn[t].fit(train_data,train_label)
    ty.append(knn[t].predict(test_data))

mae = [0 for i in range(100)]

for t in tqdm(range (100)) :
    for i in range(len(ty)) :
        mae[t] += abs(ty[t][i]-test_label[i])
    mae[t] /= len(ty)

opt = mae.index(min(mae))
print(min(mae))

output = {
    'id'            : [i+17170 for i in range (len(test_data))],
    'Danceability'  : [[0]*len(test_data)]
}

test_load = np.genfromtxt('test.csv', delimiter=',')
td = np.delete(test_load, 0, 0)

output['Danceability'] = knn[opt].predict(td)
pd.DataFrame(out).to_csv('out_knn.csv', index=False)

