#!/bin/bash
#PBS -q short-mig
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj18
#PBS -j oe

cd ${PBS_O_WORKDIR}
uv run -m sfi_utmos.train_mos fit -c configs/mos/finetune/scratch_w2v2.yaml