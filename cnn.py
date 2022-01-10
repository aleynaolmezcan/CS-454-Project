# lets import libraries that we will be usings
# we already have imported pandas and numpy
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy
import librosa
import librosa.display
import IPython.display as ipd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pickle # model pickling for future use

epohcs   = int(sys.argv[1])
batch_size     = int(sys.argv[2])

# Loading dataset
# we have 2 CSVs here, one containing features for 30 sec audio file, mean & variance for diff features we have, then
# and one for 3 sec audio files. I will be using 3 sec audio
dataf = pd.read_csv('feature_extraction/features_lr_only.csv', skiprows=1, header=None)

y = dataf.iloc[:,0] 

# scaling features
from sklearn.preprocessing import MinMaxScaler
fit = MinMaxScaler()
X = fit.fit_transform(np.array(dataf.iloc[:,1:],dtype=float))

# dividing into training and test Data
X_train,x_test, Y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)


# Using CNN algorithm
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# building model

model_ = Sequential()


# change the values of hyperparameters to find the optimal set with GridSearch
param_grid = {
              'batch_size':[128],
              'epochs' :              [100,150,200],
              #'batch_size' :          [32, 128],
              #'optimizer' :           ['Adam', 'Nadam'],
              #'dropout_rate' :        [0.2, 0.3],
              #'activation' :          ['relu', 'elu']
             }

# model.fit(X_valid, y_valid, groups=None)
model = GridSearchCV(
        estimator=model_,
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )





model.add(Dense(512,input_shape=(X_train.shape[1],),activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='accuracy')

earlystop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10,min_delta=0.0001)
modelcheck = ModelCheckpoint('best_model.hdf5',monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')

model.fit(X_train,Y_train)
#model.fit(X_train,Y_train, validation_data=(x_test,y_test), epochs=epohcs, batch_size=batch_size,verbose = 0)

# from matplotlib import pyplot 
# pyplot.plot(history.history['loss'], label='train') 
# pyplot.plot(history.history['val_loss'], label='test') 
# pyplot.legend()
# pyplot.show()

test_loss, test_accuracy = model.evaluate(x_test,y_test,batch_size=batch_size)
print("Test loss : ",test_loss)
print("\nBest test accuracy : ",test_accuracy*100)