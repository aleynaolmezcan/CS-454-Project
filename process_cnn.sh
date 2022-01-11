echo "epochs,batch_size" >> log/logs5.csv

for epochs in 100 200 400
do
    for batch_size in 32 64 128 512
    do
        for dropout in 0.1 0.2 0.3
        do 
            echo -n "${epochs},${batch_size},${dropout}" >> log/logs5.csv
            python cnn.py $epochs $batch_size $dropout  >> log/logs5.csv
        done
    done
done
