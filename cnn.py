# lets import libraries that we will be usings
# we already have imported pandas and numpy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy
import librosa
import librosa.display
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pickle # model pickling for future use

# Loading dataset
# we have 2 CSVs here, one containing features for 30 sec audio file, mean & variance for diff features we have, then
# and one for 3 sec audio files. I will be using 3 sec audio
dataf = pd.read_csv('feature_extraction/features_last.csv')
dataf.head()

dataf.tail()

dataf.shape

dataf.describe()

# removing filename column
dataf = dataf.drop(labels='song_name',axis=1)

# looking into what i have 
sample_audio = "dataset/genres/pop/pop.00003.wav"
sample, sample_rate = librosa.load(sample_audio)


ipd.Audio(sample, rate=sample_rate)

sample, sample_rate = librosa.load(sample_audio, sr=16000)
print(len(sample),sample_rate)

print(type(sample),sample_rate)
sample, sample_rate = librosa.load(sample_audio, sr=16000)

# plotting raw wave Files, here it is for Pop genre
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(211)
ax1.set_title("dataset/genres/pop/pop.00003.wav")
ax1.set_xlabel('time')
ax1.set_ylabel('Amptitude')
librosa.display.waveplot(sample)
plt.show()

# Spectrogram, we have this in images folder
# way of representing signal loudness at diff freq
# also known as Sonographs
# when data is 3D then waterfall

stft = librosa.stft(sample)
stft_db = librosa.amplitude_to_db(abs(stft))
plt.figure(figsize=(14,6))
librosa.display.specshow(stft,sr=sample_rate,x_axis='time',y_axis='hz')
plt.colorbar()


stft = librosa.stft(sample)
stft_db = librosa.amplitude_to_db(abs(stft))
plt.figure(figsize=(14,6))
librosa.display.specshow(stft_db,sr=sample_rate,x_axis='time',y_axis='hz')
plt.colorbar()

# Rolloff - feq below which a specified percentage of the total spectral lies / 85%
from sklearn.preprocessing import normalize

spectral_rolloff = librosa.feature.spectral_rolloff(sample+0.01,sr=sample_rate)[0]
plt.figure(figsize=(14,6))
librosa.display.waveplot(sample,sr=sample_rate,alpha=0.3)

# Zero crossing 
plt.figure(figsize=(14,6))
plt.plot(sample[8000:12000])
plt.grid()

#count
zero_cross = librosa.zero_crossings(sample[8000:12000],pad=False)
print("Count {}".format(sum(zero_cross)))

class_list = dataf.iloc[:,-1] 
convert = LabelEncoder()

y = convert.fit_transform(class_list)

dataf.iloc[:,:-1]

# scaling features
from sklearn.preprocessing import StandardScaler
fit = StandardScaler()
X = fit.fit_transform(np.array(dataf.iloc[:,:-1],dtype=float))

# dividing into training and test Data
X_train,x_test, Y_train, y_test = train_test_split(X,y,test_size=0.2)

print(len(Y_train),len(y_test))

# Using CNN algorithm
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# building model

model = Sequential()

model.add(Dense(512,input_shape=(X_train.shape[1],),activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))


model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='accuracy')

earlystop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10,min_delta=0.0001)
modelcheck = ModelCheckpoint('best_model.hdf5',monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')

history = model.fit(X_train,Y_train, validation_data=(x_test,y_test), epochs=600, callbacks=[earlystop,modelcheck], batch_size=128)

from matplotlib import pyplot 
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend()
pyplot.show()

test_loss, test_accuracy = model.evaluate(x_test,y_test,batch_size=128)
print("Test loss : ",test_loss)
print("\nBest test accuracy : ",test_accuracy*100)