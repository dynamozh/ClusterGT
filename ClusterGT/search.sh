#!/bin/bash

exp_name="CS"
batch_size=(128)
hidden_dim=(128)
n_layers=(2)
num_parts=(200)
k=(1)
walk_length=(10)
seed=(4407 4417 4427)


for batch_size in "${batch_size[@]}"; do
    for hidden_dim in "${hidden_dim[@]}"; do
        for n_layers in "${n_layers[@]}"; do
            for num_parts in "${num_parts[@]}"; do
                for k in "${k[@]}"; do
                    for walk_length in "${walk_length[@]}"; do
                        for seed in "${seed[@]}"; do

                            echo "=====================================ARGS======================================"
                            echo "exp_name: ${exp_name}"
                            echo "batch_size: ${batch_size}"
                            echo "hidden_dim: ${hidden_dim}"
                            echo "n_layers: ${n_layers}"
                            echo "num_parts: ${num_parts}"
                            echo "k: ${k}"
                            echo "walk_length: ${walk_length}"
                            echo "seed: ${seed}"
                            echo "==============================================================================="

                            python main.py --seed $seed --dataset_name $exp_name \
                                  --batch_size $batch_size --hidden_dim $hidden_dim\
                                  --n_layers $n_layers --num_parts $num_parts\
                                  --k $k --walk_length $walk_length\

                            wait

                        done
                    done
                done
            done
        done
    done
done
