from random import choices

print('Dont use the script. Files are already separated.')
exit()
# 0 : Training
# 1 : Testing
# 2 : Validation
population = [0, 1, 2]
weights    = [0.5, 0.25, 0.25]
distribution_samples = choices(population, weights, k=1000)

with open('features.csv', 'r') as f:
    l1 = f.readline()
    with open('training.csv', 'w+') as f_train:
        with open('testing.csv', 'w+') as f_test:
            with open('validation.csv', 'w+') as f_valid:
                f_train.write(l1)
                f_test.write(l1)
                f_valid.write(l1)
                for line in f:
                    if distribution_samples[0] == 0:
                        f_train.write(line)
                    elif distribution_samples[0] == 1:
                        f_test.write(line)
                    else:
                        f_valid.write(line)
                    distribution_samples = distribution_samples[1:]
