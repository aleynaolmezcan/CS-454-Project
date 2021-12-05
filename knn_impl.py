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


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('feature_extraction/training.csv')

X = df.drop(columns=['song_name'])  # Keeps all the features of the songs

# print(df.head())

# print(X.head())


y = df['song_name'].values  # .wav
y = calculate_target(y)  # Encoding the genres

# print(y[100:150])
#
# print(X.shape)
# print(len(y))
# print(df.shape)

pca = PCA(n_components=48)
pca.fit(X)
X = pca.transform(X)

''' Split dataset into train and test data '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)



''' Create KNN classifier '''
knn = KNeighborsClassifier(n_neighbors=15, metric='braycurtis', metric_params={'VI': np.cov(X_train)})



''' Fit the classifier to the data '''
knn.fit(X_train, y_train)

'''show first 5 model predictions on the test data '''
print('Actual : ', y_test[0:5])
print('Prediction : ', knn.predict(X_test)[0:5])

print('Score : ', knn.score(X_test, y_test))
