import librosa as lr
import numpy as np

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64 
beeg_features = []
with open('features_mels.csv', "w+") as f:
    f.write("label,mel_spectrogram\n")

    for genre in genres:
        index = genres.index(genre)
        for i in range(100):
            wav_file = '../dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            path = './dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            y, sr = lr.load(wav_file)

            melspec = lr.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=N_FFT, hop_length=HOP_LENGTH)

            print(melspec.shape)
            