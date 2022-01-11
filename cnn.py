# inspired from https://github.com/Shashabl0/Music-Genre-Classification-using-CNN/blob/master/music-genre-classification.ipynb

# lets import libraries that we will be usings
# we already have imported pandas and numpy
import sys
from keras.backend import dropout
import keras as k
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


epohcs   = int(sys.argv[1])
batch_size  = int(sys.argv[2])
dropout = float(sys.argv[3])

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
model = k.models.Sequential([
    k.layers.Dense(512,input_shape=(X_train.shape[1],),activation='relu'),
    k.layers.Dropout(dropout),

    k.layers.Dense(256, activation='relu'),
    k.layers.Dropout(dropout),

    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(dropout),

    k.layers.Dense(64, activation='relu'),
    k.layers.Dropout(dropout),

    k.layers.Dense(10, activation='softmax'),
])


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='accuracy')

#earlystop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10,min_delta=0.0001)
#modelcheck = ModelCheckpoint('best_model.hdf5',monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')

model.fit(X_train,Y_train, validation_data=(x_test,y_test), epochs=epohcs, batch_size=batch_size,verbose = 0)


# from matplotlib import pyplot 
# pyplot.plot(history.history['loss'], label='train') 
# pyplot.plot(history.history['val_loss'], label='test') 
# pyplot.legend()
# pyplot.show()

test_loss, test_accuracy = model.evaluate(x_test,y_test,batch_size=batch_size)
print("Test loss : ",test_loss)
print("Best test accuracy : ",test_accuracy*100)
print("\n\n")