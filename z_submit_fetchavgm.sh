#!/bin/bash

algorithm="fetchavgm"
job_file="z_job_fetchavgm.sh"

lr_range=(0.001 0.01 0.05 0.1 0.5 1)
lr_decay_range=(0.85 0.9 0.99)
server_momentum_range=(0.1 0.5 0.9 0.75)
batch_size_range=(10)
seed_range=(1)

for lr in "${lr_range[@]}"; do
  for lr_decay in "${lr_decay_range[@]}"; do
    for server_momentum in "${server_momentum_range[@]}"; do
      for batch_size in "${batch_size_range[@]}"; do
        for seed in "${seed_range[@]}"; do
          identifier_name="$algorithm"

          export lr
          export lr_decay
          export server_momentum
          export batch_size
          export seed

          dir="rc_out_shiv_shankar2/$identifier_name"
          mkdir -p $dir

          job_name="$algorithm-lr_${lr}-lr_dec_${lr_decay}-mom_${server_momentum}-bs_${batch_size}sd_${seed}"
          export job_name

          out_file=$dir/$job_name.out
          error_file=$dir/$job_name.err

          echo $job_name
          sbatch -J $job_name -o $out_file -e $error_file $job_file
        done
      done
    done
  done
done
