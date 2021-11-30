import csv
import pandas as pd
import numpy as np

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

training_data = pd.read_csv('./feature_extraction/training.csv').drop(['song_name'], axis=1)
print(training_data.head())

exit()
df_x = training_data
df_x['mahalanobis_dist'] = mahalanobis(x=df_x, data=training_data)
df_x.head()
