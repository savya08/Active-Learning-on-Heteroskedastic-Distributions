#! /bin/sh

datasets=(CIFAR10, SVHN)
models=(resnet)
algos=(rand, conf, marg, coreset, badge, lhd)
nqueries=(1000)
tasks=(clean random color class)
finetune=True

cd $PBS_O_WORKDIR;

for seed in 1 2 3
do
	for data in ${datasets[@]}
	do
		for model in ${models[@]}
		do
			for algo in ${algos[@]}
			do
				for nquery in ${nqueries[@]}
				do
					for task in ${tasks[@]}
					do
						exp_name=$seed-$data-$model-$algo-$nquery-$task-$finetune
						echo $exp_name
						qsub -N $exp_name -v alg=$algo,model=$model,nquery=$nquery,data=$data,mode=$task,exp_name=$exp_name,seed=$seed score-based-active-learning/submit.sh
					done
				done
			done
		done
	done
done