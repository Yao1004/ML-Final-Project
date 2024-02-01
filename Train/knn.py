from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

train_load = np.genfromtxt('train.csv', delimiter=',')
train_data = np.delete(train_load, 0, 0)
cols = len(train_data[0])
y = train_data[:, 0]
x = train_data[:, [x for x in range(1, cols)]]

train_data , test_data , train_label , test_label = train_test_split(x,y,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=50, leaf_size=2, p=10)
knn.fit(train_data,train_label)
ty = knn.predict(test_data)

mae = 0

for i in range(len(ty)) :
    mae += abs(ty[i]-test_label[i])
mae /= len(ty)

print(mae)
