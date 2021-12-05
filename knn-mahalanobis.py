import pandas as pd
import numpy as np

training_data = pd.read_csv('./feature_extraction/training.csv').drop(['song_name'], axis=1)
print(training_data.head())

cov_mat = np.cov(training_data.to_numpy(), rowvar=False)
print(cov_mat[0])
print(cov_mat.shape)


adj_matrix = np.zeros((training_data.shape[1], training_data.shape[1]))
print(adj_matrix.shape)