import sys
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.spatial.distance import pdist, wminkowski, squareform
from sklearn.model_selection import GridSearchCV   
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix, classification_report, make_scorer, f1_score, precision_score, recall_score
import seaborn as sn

used_metric     = str(sys.argv[1])
num_neighbors   = int(sys.argv[2])
pca_components  = int(sys.argv[3])
r_state         = int(sys.argv[4])

df = pd.read_csv('feature_extraction/features_last.csv')
X = df.drop(columns=['song_name'])  # Keeps all the features of the songs
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(X.iloc[:, :-1], dtype = float))


y = df['label'].values 

if pca_components > 0:
    pca = PCA(n_components=pca_components)
    pca.fit(X)
    X = pca.transform(X)

''' Split dataset into train and test data '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

#param_grid = {'used_metric' : ['mahalanobis', 'euclidean', 'minkowski', 'manhattan', 'chebyshev', 'hamming'], 'weights' : ['uniform', 'distance'],'n_neighbors': [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10] }


knn = KNeighborsClassifier(metric=used_metric, n_neighbors=num_neighbors)
# model = GridSearchCV(knn, param_grid, cv=3)

# model.fit(X_train,y_train,groups=None)



if used_metric == 'mahalanobis':
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=used_metric, metric_params={'VI': np.cov(X_train.T)})
if used_metric == 'wminkowski':
    distances = np.random.uniform(0, 1, X_train.shape[0])
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=used_metric, metric_params={'w': distances})

''' Fit the classifier to the data '''
knn.fit(X_train, y_train)
#c_mat = confusion_matrix(y_test, knn.predict(X_test))
# print(knn.score(X_val, y_val))
# print(knn.score(X_test, y_test))


test_pred = knn.predict(X_test)


#disp = ConfusionMatrixDisplay(confusion_matrix=c_mat)
#disp.plot()
#plt.show()

print("Accuracy : ", accuracy_score(y_test, test_pred))
print("Balanced Accuracy : ", balanced_accuracy_score(y_test, test_pred))
print("Matthews Correlation Coefficient: ", matthews_corrcoef(y_test, test_pred))
print("F1 Score: ", f1_score(y_test, test_pred, average = 'weighted'))
print("Precision Score: ", precision_score(y_test, test_pred, average = 'weighted'))
print("Recall Score: ", recall_score(y_test, test_pred, average = 'weighted'))

conf_matrix = confusion_matrix(y_test, test_pred)
#print('\033[1m' + "\n Table 1: Confusion Matrix of Test Data \n\n" + '\033[0m', conf_matrix)

df_cm = pd.DataFrame(conf_matrix, index = [i for i in range(10)],
                columns = [i for i in range(10)])

plt.title('Confusion Matrix KNN - Test Data')
sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
plt.savefig('./confusion_matrix_knn_test.png')
#plt.show()
plt.clf()