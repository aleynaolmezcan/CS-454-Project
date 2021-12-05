from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_target(dataset):
    temp_target = dataset[0].split("/")[-2]
    temp_value = 0
    final_array = list()
    for i in range(len(dataset)):
        name = dataset[i].split("/")[-2]
        if (name == temp_target):
            final_array.append(temp_value)
        else:
            temp_value += 1
            temp_target = dataset[i].split("/")[-2]

    return final_array


dataset = pd.read_csv('training.csv', header=None, skiprows=1)  # training dataset

# Create feature and target arrays
X = np.array(dataset.iloc[:, 1:])
y = np.array(calculate_target(dataset.iloc[:, 0].tolist()))

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
