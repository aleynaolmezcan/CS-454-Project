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


# 0 : Training
# 1 : Testing
# 2 : Validation
population = [0, 1]
weights    = [0.75, 0.25]
distribution_samples = choices(population, weights, k=1000)

with open('features_last.csv', 'r') as f:
    l1 = f.readline()
    with open('training.csv', 'w+') as f_train:
        with open('validation.csv', 'w+') as f_valid:
            f_train.write(l1)
            f_valid.write(l1)
            for line in f:
                if distribution_samples[0] == 0:
                    f_train.write(line)
                elif distribution_samples[0] == 1:
                    f_valid.write(line)
                distribution_samples = distribution_samples[1:]
