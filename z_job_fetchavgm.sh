#!/bin/bash -lT

#SBATCH --job-name=tinyimagenet

#SBATCH --account=continual
#SBATCH --partition=tier3

#SBATCH --mail-user=jt4812@g.rit.edu
#SBATCH --mail-type=ALL

#SBATCH --time 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=24g

#SBATCH --gres=gpu:a100:1

conda activate exp

python3 main.py \
  --exp_name "FedAvgM_MNIST_2NN_IID_C${c}_B${b}" --seed 42 --device cuda \
  --dataset MNIST \
  --split_type iid --test_size 0 \
  --model_name TwoNN --resize 28 --hidden_size 200 \
  --algorithm fedavgm --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
  --K 100 --R 1000 --E 1 --C $c --B $b --beta1 0.5 \
  --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_decay_step 25 --criterion CrossEntropyLoss
