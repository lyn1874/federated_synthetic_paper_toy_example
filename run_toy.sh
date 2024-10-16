#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT

# zeta=${1:-10}
# num_dim=${2:-25}
# num_nodes=${3:-10}
# version=${4:-1}
# seed_use=${5:-1024}
# b_sigma=${6:-9}
# noise_sigma=${7:-10}
# low_hession=${8:-original}

zeta=10
num_dim=25
num_nodes=10
version=6
b_sigma=0
noise_sigma=0
low_hession=original

seed_use=12988

for i in $(seq 0 1 30)
do 
    python3 sec_toy_exp.py --zeta "$zeta" --num_dim "$num_dim" --num_nodes "$num_nodes" --version "$version" \
        --gamma "$i" --seed_use "$seed_use" --b_sigma "$b_sigma" --noise_sigma "$noise_sigma" --low_hession "$low_hession" & 
done 

wait 

