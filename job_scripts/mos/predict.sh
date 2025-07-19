#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -P gcb50354
#PBS -j oe
#PBS -k oed
cd ${PBS_O_WORKDIR}
uv run notebooks/predict.py --ckpt_path /home/acc12576tt/github.com/sfi-utmos/notebooks/../vmc24_mos_v2/0nfp30qu/checkpoints/epoch=09-val_loss=0.00.ckpt --id 0nfp30qu --wav_dir /home/acc12576tt/github.com/sfi-utmos/audiomos2025-track3-eval-phase/DATA/wav &
uv run notebooks/predict.py --ckpt_path /home/acc12576tt/github.com/sfi-utmos/notebooks/../vmc24_mos_v2/8evq2olc/checkpoints/epoch=09-val_loss=0.00.ckpt --id 8evq2olc --wav_dir /home/acc12576tt/github.com/sfi-utmos/audiomos2025-track3-eval-phase/DATA/wav &
uv run notebooks/predict.py --ckpt_path /home/acc12576tt/github.com/sfi-utmos/notebooks/../vmc24_mos_v2/iy580i6j/checkpoints/epoch=15-val_loss=0.00.ckpt --id iy580i6j --wav_dir /home/acc12576tt/github.com/sfi-utmos/audiomos2025-track3-eval-phase/DATA/wav &
uv run notebooks/predict.py --ckpt_path /home/acc12576tt/github.com/sfi-utmos/notebooks/../vmc24_mos_v2/k3fudi6p/checkpoints/epoch=15-val_loss=0.00.ckpt --id k3fudi6p --wav_dir /home/acc12576tt/github.com/sfi-utmos/audiomos2025-track3-eval-phase/DATA/wav &
uv run notebooks/predict.py --ckpt_path /home/acc12576tt/github.com/sfi-utmos/notebooks/../vmc24_mos_v2/lifs4g77/checkpoints/epoch=07-val_loss=0.00.ckpt --id lifs4g77 --wav_dir /home/acc12576tt/github.com/sfi-utmos/audiomos2025-track3-eval-phase/DATA/wav &
uv run notebooks/predict.py --ckpt_path /home/acc12576tt/github.com/sfi-utmos/notebooks/../vmc24_mos_v2/s9y04uck/checkpoints/epoch=09-val_loss=0.00.ckpt --id s9y04uck --wav_dir /home/acc12576tt/github.com/sfi-utmos/audiomos2025-track3-eval-phase/DATA/wav &
uv run notebooks/predict.py --ckpt_path /home/acc12576tt/github.com/sfi-utmos/notebooks/../vmc24_mos_v2/ulziex13/checkpoints/epoch=13-val_loss=0.00.ckpt --id ulziex13 --wav_dir /home/acc12576tt/github.com/sfi-utmos/audiomos2025-track3-eval-phase/DATA/wav &
wait