#!/bin/bash
#
set -e
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for GPU-ID"
  exit 1 
fi
gpus=$1
dataset=AWA2
clusters=3
r_net=IPNX-256-2048-40-10-2
c_net=Linear-2048
loss_type=softmax-100-0
class_per_it=30
num_shot=1
lr=0.00002
weight_decay=0.0001
epochs=360
consistency_type=kla2i
consistency_coef=1
manual_seed=26961

zshot_dir="${HOME}/datasets/zshots"
save_dir=./logs/NOEP-IPN-${dataset}/Consistency.${consistency_type}.${consistency_coef}-C${clusters}-${loss_type}-R-${r_net}.C-${c_net}-N${class_per_it}-K${num_shot}-LR${lr}-WD${weight_decay}

CUDA_VISIBLE_DEVICES=${gpus} python exps/main-no-episode.py \
	       --consistency_coef ${consistency_coef} --dataset ${dataset} \
	       --consistency_type ${consistency_type} \
	       --data_root ${zshot_dir}/info-files/x-${dataset}-data-image.pth \
	       --log_dir ${save_dir} --loss_type ${loss_type} --clusters ${clusters} \
	       --relation_name ${r_net} --semantic_name ${c_net} \
	       --class_per_it ${class_per_it} --num_shot ${num_shot} \
	       --epochs ${epochs} --lr ${lr} \
	       --weight_decay ${weight_decay} \
	       --log_interval 50 --test_interval 1 \
	       --manual_seed ${manual_seed}
