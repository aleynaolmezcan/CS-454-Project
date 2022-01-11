echo "Metrics,Num_Neighbors,PCA_Components,N_Random_States,Score" > log/logs8.csv

metrics="minkowski mahalanobis euclidean manhattan"

for metric in $metrics
do
    for neighbor in 1 2 3 4 5 6 7 8 9 10 11 13 15 20
    do
        for PCA_Components in 0 3 4 5 6 7 8 9 10 13 15 20 25 30 35 40 45 50 55 60 65 70 75
        do
            for state in 1 2 3 5 10 15 20
            do
                echo -n "${metric},${neighbor},${PCA_Components},${state}," >> log/logs8.csv
                python knn_impl.py $metric $neighbor $PCA_Components $state >> log/logs8.csv
            done
        done
    done
done