from random import choices

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

print('Dont use the script. Files are already separated.')
with open('features_sr_fixed.csv', 'r+') as f:
    with open('features_last.csv', 'w+') as g:
        g.write(f.readline().strip() + ",label\n")
        for line in f:
            line = line.strip()
            song_name = line.split(',')[0]
            label = song_name.split('/')[-2]
            line += "," + str(genres.index(label)) + "\n"
            g.write(line)
