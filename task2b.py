import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
import itertools
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
import itertools
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from itertools import combinations 
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer


world = pd.read_csv('world.csv')
life = pd.read_csv('life.csv')

merged = pd.merge(world, life,on = 'Country Code', how = 'right',sort=True)
merged.drop('Country', inplace=True, axis=1)
merged.drop('Year', inplace = True, axis = 1)
merged.drop('Country Code', inplace = True, axis = 1)
merged.drop('Time', inplace = True, axis = 1)
merged.drop('Country Name', inplace = True, axis = 1)
merged.columns = range(merged.shape[1])

# dataframe without the class label of life expectancy
#new_merged = merged.copy(deep=True)
#new_merged = new_merged.drop([new_merged.columns[20]] ,  axis='columns')

data = merged.iloc[:, 0:20]
classlabel=merged.iloc[:, 20]
#train and impute the data
X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=0.70, test_size=0.30, random_state = 200)
X_train = X_train.replace('..', np.nan)
X_test = X_test.replace('..', np.nan)


imp_median= SimpleImputer(missing_values=np.nan, strategy='median')
imp_median = imp_median.fit(X_train)
X_train =imp_median.transform(X_train)
X_test =imp_median.transform(X_test)

#imputed training set
training=pd.DataFrame(data=X_train[0:,0:] ,columns=[i for i in range(data.shape[1])])

#imputed training set forming new features set
new = pd.DataFrame({'{}-{}'.format(a, b): training[a] * training[b] for a, b in itertools.combinations(training.columns, 2)})
feature_train = pd.concat([training, new], axis=1)

testing=pd.DataFrame(data=X_test[0:,0:] ,columns=[i for i in range(data.shape[1])])
#imputed training set forming new features set
new_test = pd.DataFrame({'{}-{}'.format(a, b): testing[a] * testing[b] for a, b in itertools.combinations(testing.columns, 2)})
feature_test = pd.concat([testing, new_test], axis=1)

X1 = StandardScaler().fit_transform(training)
sum_of_squared_distances = []
K = range(1,15)
for k in K:
    k_means = KMeans(n_clusters=k)
    model = k_means.fit(X1)
    sum_of_squared_distances.append(k_means.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
plt.title('elbow method for optimal k')
plt.tight_layout()
plt.savefig('task2graph1.png')
plt.close()


calinski= []
silhouette = []
lst = [i for i in range(2,15)]
for i in range(2,15):
    k_means = KMeans(n_clusters=i)
    model = k_means.fit(X1)
    y_hat = k_means.predict(X1)
    labels = k_means.labels_
    silhouette.append(metrics.silhouette_score(X1, labels, metric = 'euclidean'))
    calinski.append(metrics.calinski_harabasz_score(X1, labels))
    
silhouette = preprocessing.scale(silhouette)
calinski =  preprocessing.scale(calinski)
best_k = pd.DataFrame({'k':lst,'calinski':calinski,'silhouette': silhouette})

best_k.to_csv('best_k.csv', index=False)

best_k_data = pd.read_csv('best_k.csv')
best_k_data['mean'] = best_k_data.iloc[:,1::].mean(axis=1)
index = (best_k_data.iloc[:,-1].idxmax())
k_clusters = best_k_data.iloc[index,0]

k_means = KMeans(n_clusters=k_clusters).fit(X1)
centers = k_means.cluster_centers_
labels = k_means.labels_

labels = labels.tolist()

labels_data = pd.DataFrame({'clusterlabel': labels})

df = pd.concat([feature_train, labels_data], axis=1)

X = df.iloc[:,:]  #independent columns
y = y_train   #target column i.e price range

#apply SelectKBest class to extract top 4 best features
bestfeatures = SelectKBest(score_func=f_classif, k=4)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores = featureScores.nlargest(4,'Score')
list_of_feature = featureScores['Specs'].to_list()

data = (df[list_of_feature])

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(data, y_train)

y_pred=knn.predict(feature_test[list_of_feature])

first_four = df.iloc[:,0:4]
list_of_feat = [0, 1, 2, 3]
knn1 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn1.fit(first_four, y_train)

y1_pred=knn1.predict(feature_test[list_of_feat])


from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(),
                    PCA(n_components=4))

knn2 = neighbors.KNeighborsClassifier(n_neighbors=3)

model.fit(feature_train, y_train)
    # Fit a nearest neighbor classifier on the embedded training set
knn2.fit(model.transform(feature_train), y_train)

acc_knn = knn2.score(model.transform(feature_test), y_test)

print("Accuracy of feature engineering:",round(accuracy_score(y_test, y_pred),3))
print("Accuracy of PCA:",round(acc_knn,3))
print("Accuracy of first four features:",round(accuracy_score(y_test, y1_pred),3))
