#!/bin/sh

c=0.1
e=1 #5

# FedSGD experiments in Table 1 of (McMahan et al., 2016)
## IID split
for b in 0 10; do
  python3 main.py \
    --exp_name "FedAdam_MNIST_2NN_IID_C${c}_B{$b}" --seed 42 --device cuda \
    --dataset MNIST \
    --split_type iid --test_size 0 \
    --model_name TwoNN --resize 28 --hidden_size 200 \
    --algorithm fedadam --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 1000 --E $e --C $c --B $b --beta1 0.9 --beta2 0.99 --tau 1e-6 --server_lr 0.01 \
    --optimizer SGD --lr 1.0 --lr_decay 0.99 --lr_decay_step 25 --criterion CrossEntropyLoss
done
## Pathological Non-IID split
#c=0.1
#python3 main.py \
#  --exp_name "FedAdam_MNIST_2NN_Patho_C${c}_B0" --seed 42 --device cuda \
#  --dataset MNIST \
#  --split_type patho --test_size 0 \
#  --model_name TwoNN --resize 28 --hidden_size 200 \
#  --algorithm fedadam --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
#  --K 100 --R 1000 --C $c --B 0 --beta1 0 \
#  --optimizer SGD --lr 0.1 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss
