import sys
import pandas as pd
import numpy as np
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.distance import pdist, wminkowski, squareform


used_metric     = str(sys.argv[1])
num_neighbors   = int(sys.argv[2])
pca_components  = int(sys.argv[3])
r_states        = int(sys.argv[4])


df = pd.read_csv('feature_extraction/features_last.csv')
X = df.drop(columns=['song_name'])  # Keeps all the features of the songs
scaler = StandardScaler()
X = scaler.fit_transform(np.array(X.iloc[:, :-1], dtype = float))


y = df['label'].values 

if pca_components > 0:
    pca = PCA(n_components=pca_components)
    pca.fit(X)
    X = pca.transform(X)

''' Split dataset into train and test data '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=r_states, stratify=y)

knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=used_metric)

if used_metric == 'mahalanobis':
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=used_metric, metric_params={'VI': np.cov(X_train.T)})
if used_metric == 'wminkowski':
    distances = np.random.uniform(0, 1, X_train.shape[0])
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=used_metric, metric_params={'w': distances})

''' Fit the classifier to the data '''
knn.fit(X_train, y_train)
#c_mat = confusion_matrix(y_test, knn.predict(X_test))
print(knn.score(X_test, y_test))

#disp = ConfusionMatrixDisplay(confusion_matrix=c_mat)
#disp.plot()
#plt.show()

''' Predict the response for test dataset '''