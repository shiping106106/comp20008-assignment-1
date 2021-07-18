import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

world = pd.read_csv('world.csv')
life = pd.read_csv('life.csv')

merged = pd.merge(world, life,on = 'Country Code', how = 'right',sort=True)

merged.drop('Country', inplace=True, axis=1)
merged.drop('Year', inplace = True, axis = 1)
merged.drop('Country Code', inplace = True, axis = 1)
merged.drop('Time', inplace = True, axis = 1)
merged.drop('Country Name', inplace = True, axis = 1)

data_top = merged.head()  
lst = [i for i in data_top.columns]
merged.columns = range(merged.shape[1])

data = merged.iloc[:, 0:20]
classlabel=merged.iloc[:, 20]


X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=0.70, test_size=0.30, random_state=200)
X_train = X_train.replace('..', np.nan)
X_test = X_test.replace('..', np.nan)

imp_median= SimpleImputer(missing_values=np.nan, strategy='median')
imp_median = imp_median.fit(X_train)
X_train =imp_median.transform(X_train)
X_test =imp_median.transform(X_test)

training=pd.DataFrame(data=X_train[0:,0:] ,columns=[i for i in range(data.shape[1])])

mean = [round(training[i].mean(),3) for i in range(training.shape[1])]
median = [round(training[i].median(),3) for i in range(training.shape[1])]
var = [round(training[i].var(),3) for i in range(training.shape[1])]

scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)

knn1 = neighbors.KNeighborsClassifier(n_neighbors=7)
knn1.fit(X_train, y_train)

y1_pred=knn1.predict(X_test)

dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train, y_train)
pred=dt.predict(X_test)

print("Accuracy of decision tree:",round(accuracy_score(y_test, pred),3))
print("Accuracy of k-nn (k=3):",round(accuracy_score(y_test, y_pred),3))
print("Accuracy of k-nn (k=7):",round(accuracy_score(y_test, y1_pred),3))


df = pd.DataFrame({'feature':lst[0:-1], 'median':median, 'mean':mean, 'variance':var})
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
  
df.to_csv('task2a.csv', index=False)

