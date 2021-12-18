import numpy as np
from tensorflow.python import tf2
import keras.models
from keras.models import Sequential
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('feature_extraction/features_last.csv')

df.drop(labels='song_name', axis=1)
# X = df.drop(columns=['song_name'])  # Keeps all the features of the songs
X = df.drop(columns=['label', 'song_name'])

y = df['label'].values

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
result_list = np.zeros(y.shape)

for i in y:
    np.append(result_list, genres.index(str(i)))

y = result_list

''' Split dataset into train and test data '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2, stratify=y)


def trainModel(model, epochs, optimizer):
    batch_size = 32
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy'
                  )
    return model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )


def plotValidate(history):
    print("Validation Accuracy", max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12, 6))
    plt.show()


model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(10, activation='softmax'),
])

print(model.summary())
model_history = trainModel(model=model, epochs=100, optimizer='adam')

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=32)
print('\ntest loss: ', test_loss, '\ntest accuracy: ', test_accuracy)
