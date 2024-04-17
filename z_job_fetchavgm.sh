#!/bin/bash -lT

#SBATCH --job-name=tinyimagenet

#SBATCH --account=continual
#SBATCH --partition=tier3

#SBATCH --mail-user=jt4812@g.rit.edu
#SBATCH --mail-type=ALL

#SBATCH --time 0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=10g

#SBATCH --gres=gpu:a100:1

conda activate fl

python3 main.py \
  --exp_name $job_name --seed 42 --device cuda \
  --dataset MNIST \
  --split_type iid --test_size 0 \
  --model_name TwoNN --resize 28 --hidden_size 200 \
  --algorithm fedavgm --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
  --K 100 --R 1000 --E 1 --C 0.1 --B $batch_size \
  --lr_decay_step 25 --criterion CrossEntropyLoss \
  --optimizer SGD --beta1 $server_momentum --lr $lr --lr_decay $lr_decay
