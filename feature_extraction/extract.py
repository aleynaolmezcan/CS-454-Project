import librosa as lr
import numpy as np

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

beeg_features = []
with open('features.csv', "w+") as f:
    f.write("song_name,spectral_rollooff,spectral_centroid,spectral_bandwidth,spectral_flatness,spectral_contrast,tempogram,fourier_tempogram,tempo,rms,zero_crossing_rate,chroma_stft,chroma_cens,chroma_cqt")
    for i in range(20):
        f.write(",mfcc_" + str(i))
    f.write('\n')
    for genre in genres:
        for i in range(100):
            wav_file = '../dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            path = './dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            y, sr = lr.load(wav_file, sr=None)

            # Extract Magnitude Based (timbral) features from the audio files
            # i.e. spectral rolloff, flux, centroid, spread, decrease, slope, 
            # flatness, and MFCCs
            f_spectral_rolloff  = lr.feature.spectral_rolloff(y=y, sr=sr)
            #spectral_flux
            f_spectral_centroid = lr.feature.spectral_centroid(y=y, sr=sr)
            f_spectral_bandwidth = lr.feature.spectral_bandwidth(y=y, sr=sr)
            #spectral_spread   = lr.feature.spectral_spread(x, sr=sr)
            #spectral_decrease
            #spectral_slope
            f_spectral_flatness = lr.feature.spectral_flatness(y=y)
            f_spectral_contrast = lr.feature.spectral_contrast(y=y, sr=sr)
            f_mfccs = lr.feature.mfcc(y=y, sr=sr)

            # Extract Tempo Based Features from the audio files
            # i.e. BPM, Energy using RMS, and beat histogram
            f_tempogram         = lr.feature.tempogram(y=y, sr=sr)
            f_fourier_tempogram = lr.feature.fourier_tempogram(y=y, sr=sr)
            f_tempo             = lr.beat.tempo(y=y, sr=sr)
            f_rms               = lr.feature.rms(y=y)

            # Extract Pitch Based Features from the audio files
            # i.e. zero crossing rate
            f_zero_crossing_rate = lr.feature.zero_crossing_rate(y=y)

            # Extract Chordal Progression Features from the audio files
            # i.e. chroma
            f_chroma_stft = lr.feature.chroma_stft(y=y, sr=sr)
            f_chroma_cens = lr.feature.chroma_cens(y=y, sr=sr)
            f_chroma_cqt = lr.feature.chroma_cqt(y=y, sr=sr)

            f.write(f'{path},{np.mean(f_spectral_rolloff)},{np.mean(f_spectral_centroid)},{np.mean(f_spectral_bandwidth)},{np.mean(f_spectral_flatness)},{np.mean(f_spectral_contrast)},{np.mean(f_tempogram)},{np.mean(f_tempo)},{np.mean(f_rms)},{np.mean(f_zero_crossing_rate)},{np.mean(f_chroma_stft)},{np.mean(f_chroma_cens)},{np.mean(f_chroma_cqt)}')
            #print  (f'{path},{np.mean(f_spectral_rolloff)},{np.mean(f_spectral_centroid)},{np.mean(f_spectral_bandwidth)},{np.mean(f_spectral_flatness)},{np.mean(f_spectral_contrast)},{np.mean(f_tempogram)},{np.mean(f_tempo)},{np.mean(f_rms)},{np.mean(f_zero_crossing_rate)},{np.mean(f_chroma_stft)},{np.mean(f_chroma_cens)},{np.mean(f_chroma_cqt)}',end='')
            for i in f_mfccs:
                f.write(',' + str(np.mean(i)))
                #print(',' + str(np.mean(i)), end='')
            f.write('\n')
            #print()
            # Save features to np file
            #np.save('../feature_extraction/' + song_index + '.npy', features)

            # Write features to csv file
