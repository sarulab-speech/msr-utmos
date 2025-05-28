#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -P gcb50354
#PBS -j oe
#PBS -k oed
cd ${PBS_O_WORKDIR}
uv run -m sfi_utmos.train_mos fit -c configs/mos/pretrain/pretrain_hubert.yaml