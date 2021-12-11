import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from umap import UMAP
import umap


sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

df = pd.read_csv("feature_extraction/features_last.csv")
#print(df.head())

X = df.drop(columns=['song_name', 'label'])
y = df['label'].values
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
z = []

for i in range(1000):
    z.append(genres.index(y[i]))
     
y = np.array(z)

print(y)
print(y.shape)
df1 = pd.DataFrame({'label': y})
X.join(df1)
print(df1.head())

reducer = umap.UMAP(random_state=42)
reducer.fit(X,df1)

UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='mahalanobis',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='mahalanobis',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)

embedding = reducer.transform(X)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
print(embedding.shape)

plt.scatter(embedding[:, 0], embedding[:, 1], c = df1.label, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24)

plt.show()