echo "epochs,batch_size" >> log/logs5.csv

for epochs in 50 100 150 200 500 750 1000
do
    for batch_size in 8 16 32 64 128 512 1024 2048
    do
        echo -n "${epochs},${batch_size}" >> log/logs5.csv
        python cnn.py $epochs $batch_size  >> log/logs5.csv
    done
done
