#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -P gcb50354
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}
N_SHARDS=7
for i in $(seq 0 $((N_SHARDS - 1))); do
    echo "Starting shard $i"
    uv run -m sfi_utmos.train_mos fit -c configs/mos/finetune_condition_sr/scratch_w2v2.yaml \
        --data.train_mos_data_path="notebooks/train_${i}_7fold.csv" \
        --data.valid_mos_data_path="notebooks/val_${i}_7fold.csv" --trainer.logger.name="scratch_condition_sr_w2v2/fold${i}" & 
done
wait