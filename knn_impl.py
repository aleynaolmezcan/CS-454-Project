import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric

def calculate_target(dataset):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    result_list = list()

    for i in range(dataset.shape[0]):
        result_list.append(genres.index(dataset[i].split('/')[-2]))

    return result_list


used_metric     = str(sys.argv[1])
num_neighbors   = int(sys.argv[2])
pca_components  = int(sys.argv[3])
r_states        = int(sys.argv[4])


df = pd.read_csv('feature_extraction/training.csv')
X = df.drop(columns=['song_name'])  # Keeps all the features of the songs

y = df['song_name'].values 
y = calculate_target(y)

if pca_components > 0:
    pca = PCA(n_components=pca_components)
    pca.fit(X)
    X = pca.transform(X)

''' Split dataset into train and test data '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=r_states, stratify=y)

knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=used_metric)
if used_metric == 'mahalanobis':
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=used_metric, metric_params={'VI': np.cov(X_train.T)})


''' Fit the classifier to the data '''
knn.fit(X_train, y_train)

''' Predict the response for test dataset '''
print(knn.score(X_test, y_test))
