import librosa as lr
from librosa.filters import mel
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64 
beeg_features = []


import numpy as np
import librosa as lb 
import matplotlib.pyplot as plt

SR = 22050
N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64 

def log_melspectrogram(data, log=True, plot=False, num='', genre=""):

	melspec = lb.feature.melspectrogram(y=data, hop_length = HOP_LENGTH, n_fft = N_FFT, n_mels = N_MELS)

	if log:
		melspec = lb.power_to_db(melspec**2)

	if plot:
		melspec = melspec[np.newaxis, :]
		plt.imshow(melspec.reshape((melspec.shape[1],melspec.shape[2])))
		plt.savefig('melspec'+str(num)+'_'+str(genre)+'.png')

	return melspec

def batch_log_melspectrogram(data_list, log=True, plot=False):
	melspecs = np.asarray([log_melspectrogram(data_list[i],log=log,plot=plot) for i in range(len(data_list))])
	#this line may or may not be neccesary idk
	# melspecs = melspecs.reshape(melspecs.shape[0], melspecs.shape[1], melspecs.shape[2], 1)
	return melspecs



mels = np.zeros(16576, dtype = np.float32)
labels = np.array

with open('features_mels.csv', "w+") as f:
    f.write("label,mel_spectrogram\n")

    for genre in genres:
        index = genres.index(genre)
        for i in range(1):
            
            wav_file = '../dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            y, sr = lr.load(wav_file, offset=5.0, duration=3.0)
            
            labels = np.append(labels,wav_file)

            melspec    = lr.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            stft_graph = lr.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)

            melspec = melspec.flatten()
            melspec = melspec.reshape(1,melspec.shape[0])
            print(melspec.shape)
            np.savetxt(f, melspec, delimiter="\n", newline = ",")
            #mels = np.concatenate((mels, melspec), axis=0)


    
