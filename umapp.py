import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from mpl_toolkits.mplot3d import Axes3D

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

df = pd.read_csv("feature_extraction/features_last.csv")
#print(df.head())

X = df.drop(columns=['song_name', 'label'])
y = df['label'].values



df1 = pd.DataFrame({'label': y})
data = X.join(df1)
print(data.head())

reducer = umap.UMAP(metric='wminkowski',
     min_dist=0.01, n_components=3,
     n_neighbors=15)

reducer.fit(X, df1)

embedding = reducer.transform(X)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
print(embedding.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = df1.label, cmap= 'tab10')
plt.title('UMAP projection of the GTZAN dataset', fontsize=24)

plt.show()