#! /bin/sh

#PBS -q dgx
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=24:00:00

python3 run.py \
--alg $alg \
--model $model \
--nQuery $nquery \
--data $data \
--nStart 2000 \
--nEnd 12000 \
--finetune \
--mult 4 \
--mode $mode \
--save experiment_results/$exp_name/$seed/