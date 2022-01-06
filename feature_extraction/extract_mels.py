import numpy as np
import librosa as lb 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64

def log_melspectrogram(data, log=True, plot=False, num='', genre=""):

    melspec = lb.feature.melspectrogram(y=data, hop_length = HOP_LENGTH, n_fft = N_FFT, n_mels = N_MELS, dtype=np.float32)

    if log:
        melspec = lb.power_to_db(melspec**2)

    if plot:
        melspec = melspec[np.newaxis, :]
        plt.imshow(melspec.reshape((melspec.shape[1],melspec.shape[2])))
        plt.savefig('./mel_pics/melspec'+str(num)+'_'+str(genre)+'.png')

    return melspec

def batch_log_melspectrogram(data_list, log=True, plot=False):
    melspecs = np.asarray([log_melspectrogram(data_list[i],log=log,plot=plot) for i in range(len(data_list))])
    #this line may or may not be neccesary idk
    # melspecs = melspecs.reshape(melspecs.shape[0], melspecs.shape[1], melspecs.shape[2], 1)
    return melspecs



with open('features_mels.csv', "w+") as f:
    f.write("label,mel_spectrogram\n")

    for genre in genres:
        index = genres.index(genre)
        for i in range(100):
            
            wav_file = '../dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            y, sr = lb.load(wav_file, sr=22050, offset=5.0, duration=2.5, mono=True)
            

            melspec    = log_melspectrogram(y, log=True, plot=True, num=i, genre=genre)

            print(melspec.shape)
            f.write(f'{index},')
            melspec = melspec.flatten()
            melspec = melspec.reshape(1,melspec.shape[0])
            np.savetxt(f, melspec, delimiter=",", newline = "\n", fmt='%1.5f')


df = pd.read_csv('features_mels.csv', skiprows=1, header=None)
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled).to_csv('features_mels_scaled.csv', index=False, header=False)