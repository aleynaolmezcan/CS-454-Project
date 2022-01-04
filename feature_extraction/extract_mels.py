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

labelsWithMelspec = np.array

with open('features_mels.csv', "w+") as f:
    f.write("label,mel_spectrogram\n")

    for genre in genres:
        index = genres.index(genre)
        for i in range(1):
            
            wav_file = '../dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            path = './dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            y, sr = lr.load(wav_file, offset=5.0, duration=3.0)

            melspec    = lr.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            stft_graph = lr.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)

            melspec = melspec.flatten()
            output = np.array(wav_file)
            output = np.append(output,melspec)
            labelsWithMelspec = np.append(labelsWithMelspec,output)

np.savez("mels",labelsWithMelspec)