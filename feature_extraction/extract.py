import librosa as lr
import numpy as np

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for genre in genres:
    for i in range(100):
        wav_file = '../dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
        x, sr = lr.load(wav_file, sr=None)

        # Extract Magnitude Based (timbral) features from the audio files
        # i.e. spectral rolloff, flux, centroid, spread, decrease, slope, 
        # flatness, and MFCCs
        spectral_rolloff  = lr.feature.spectral_rolloff(x, sr=sr)
        #spectral_flux
        spectral_centroid = lr.feature.spectral_centroid(x, sr=sr)
        #spectral_spread   = lr.feature.spectral_spread(x, sr=sr)
        #spectral_decrease
        #spectral_slope
        spectral_flatness = lr.feature.spectral_flatness(x)
        mfccs = lr.feature.mfcc(x, sr=sr)
        mfccs_2 = lr.feature.mfcc(x, sr=sr, n_mfcc=2)
        mfccs_3 = lr.feature.mfcc(x, sr=sr, n_mfcc=3)
        mfccs_4 = lr.feature.mfcc(x, sr=sr, n_mfcc=4)
        mfccs_5 = lr.feature.mfcc(x, sr=sr, n_mfcc=5)


        # Extract Tempo Based Features from the audio files
        # i.e. BPM, Energy using RMS, and beat histogram
        tempo = lr.tempo(x, sr=sr)



        # Extract Pitch Based Features from the audio files
        # i.e. zero crossing rate

        zero_crossing_rate = lr.feature.zero_crossing_rate(x)

            
        # Extract Chordal Progression Features from the audio files
        # i.e. chroma
        chroma_stft = lr.feature.chroma_stft(x, sr=sr)
        chroma_cens = lr.feature.chroma_cens(x, sr=sr)
        chroma_cqt = lr.feature.chroma_cqt(x, sr=sr)










