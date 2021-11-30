import librosa as lr
import numpy as np

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

beeg_features = []
with open('features_new.csv', "w+") as f:
    f.write("song_name,mean_spectral_rolloff,std_spectral_rolloff,mean_spectral_centroid,std_spectral_centroid,mean_spectral_bandwidth,std_spectral_bandwidth"
    + ",mean_spectral_flatness,std_spectral_flatness,mean_spectral_contrast,std_spectral_contrast"
    + ",mean_tempogram,std_tempogram,mean_tempo,std_tempo,mean_rms,std_rms,mean_zero_crossing_rate,std_zero_crossing_rate"
    + ",mean_chroma_stft,std_chroma_stft,mean_chroma_cens,std_chroma_cens,mean_chroma_cqt,std_chroma_cqt"
    + ",mean_fifth_x_axis,std_fifth_x_axis,mean_fifth_y_axis,std_fifth_y_axis,mean_minor_x_axis,std_minor_x_axis,mean_minor_y_axis,std_minor_y_axis"
    + ",mean_major_x_axis,std_major_x_axis,mean_major_y_axis,std_major_y_axis")
    for i in range(20):
        f.write(",mean_mfcc_" + str(i))
        f.write(",std_mfcc_" + str(i))

    f.write('\n')

    for genre in genres:
        for i in range(100):
            wav_file = '../dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            path = './dataset/genres/' + genre + '/' + genre + '.' + f'{i:0>5}' + '.wav'
            y, sr = lr.load(wav_file, sr=None)

            # Extract Magnitude Based (timbral) features from the audio files
            # i.e. spectral rolloff, flux, centroid, spread, decrease, slope, 
            # flatness, and MFCCs
            f_spectral_rolloff    = lr.feature.spectral_rolloff(y=y, sr=sr)
            #spectral_flux
            f_spectral_centroid   = lr.feature.spectral_centroid(y=y, sr=sr)
            f_spectral_bandwidth  = lr.feature.spectral_bandwidth(y=y, sr=sr)
            #spectral_spread      = lr.feature.spectral_spread(x, sr=sr)
            #spectral_decrease
            #spectral_slope
            f_spectral_flatness = lr.feature.spectral_flatness(y=y)
            f_spectral_contrast = lr.feature.spectral_contrast(y=y, sr=sr)
            f_mfccs             = lr.feature.mfcc(y=y, sr=sr)

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
            f_chroma_stft  = lr.feature.chroma_stft(y=y, sr=sr)
            f_chroma_cens  = lr.feature.chroma_cens(y=y, sr=sr)
            f_chroma_cqt   = lr.feature.chroma_cqt(y=y, sr=sr)

            f_tonnetz      = lr.feature.tonnetz(y=y, sr=sr)


            f.write(f'{path},{np.mean(f_spectral_rolloff)},{np.std(f_spectral_rolloff)},{np.mean(f_spectral_centroid)},{np.std(f_spectral_centroid)},{np.mean(f_spectral_bandwidth)},{np.std(f_spectral_bandwidth)}')
            f.write(f',{np.mean(f_spectral_flatness)},{np.std(f_spectral_flatness)},{np.mean(f_spectral_contrast)},{np.std(f_spectral_contrast)}')
            f.write(f',{np.mean(f_tempogram)},{np.std(f_tempogram)},{np.mean(f_tempo)},{np.std(f_tempo)},{np.mean(f_rms)},{np.std(f_rms)},{np.mean(f_zero_crossing_rate)},{np.std(f_zero_crossing_rate)}')
            f.write(f',{np.mean(f_chroma_stft)},{np.std(f_chroma_stft)},{np.mean(f_chroma_cens)},{np.std(f_chroma_cens)},{np.mean(f_chroma_cqt)},{np.std(f_chroma_cqt)}')
            f.write(f',{np.mean(f_tonnetz[0])},{np.std(f_tonnetz[0])},{np.mean(f_tonnetz[1])},{np.std(f_tonnetz[1])},{np.mean(f_tonnetz[2])},{np.std(f_tonnetz[2])},{np.mean(f_tonnetz[3])},{np.std(f_tonnetz[3])}')
            f.write(f',{np.mean(f_tonnetz[4])},{np.std(f_tonnetz[4])},{np.mean(f_tonnetz[5])},{np.std(f_tonnetz[5])}')
            

            for i in f_mfccs:
                f.write(',' + str(np.mean(i)))
                f.write(',' + str(np.std(i)))
            
            f.write('\n')