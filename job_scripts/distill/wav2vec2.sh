
#!/bin/bash
#PBS -q regular-g 
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -W group_list=gj18
#PBS -j oe

cd ${PBS_O_WORKDIR}
uv run python -m  sfi_utmos.train fit -c configs/distill/config_w2v2.yaml
