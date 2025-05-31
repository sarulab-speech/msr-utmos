#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -P gcb50354
#PBS -j oe
#PBS -k oed
cd ${PBS_O_WORKDIR}
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/1p0dsya1/checkpoints/epoch=03-val_loss=0.00.ckpt --id 1p0dsya1 &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/1vsimmt6/checkpoints/epoch=02-val_loss=0.00.ckpt --id 1vsimmt6 &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/nnff2a2n/checkpoints/epoch=02-val_loss=0.00.ckpt --id nnff2a2n &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/qu54ezqe/checkpoints/epoch=04-val_loss=0.00.ckpt --id qu54ezqe &
wait
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/r22lsl0b/checkpoints/epoch=10-val_loss=0.00.ckpt --id r22lsl0b &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/rj9ygf0t/checkpoints/epoch=12-val_loss=0.00.ckpt --id rj9ygf0t &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/wbv0w6sd/checkpoints/epoch=04-val_loss=0.00.ckpt --id wbv0w6sd &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/cock94yw/checkpoints/epoch=13-val_loss=0.00.ckpt --id cock94yw &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/fyncc5cb/checkpoints/epoch=04-val_loss=0.00.ckpt --id fyncc5cb &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/iskrhoan/checkpoints/epoch=02-val_loss=0.00.ckpt --id iskrhoan &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/r31rrm4n/checkpoints/epoch=11-val_loss=0.00.ckpt --id r31rrm4n &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/sjdfgh9h/checkpoints/epoch=05-val_loss=0.00.ckpt --id sjdfgh9h &
wait
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/tkod1wmw/checkpoints/epoch=06-val_loss=0.00.ckpt --id tkod1wmw &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/y2o46zpu/checkpoints/epoch=07-val_loss=0.00.ckpt --id y2o46zpu &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/1alxuhvh/checkpoints/epoch=02-val_loss=0.00.ckpt --id 1alxuhvh &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/7xt1dj3x/checkpoints/epoch=04-val_loss=0.00.ckpt --id 7xt1dj3x &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/9za5k16l/checkpoints/epoch=04-val_loss=0.00.ckpt --id 9za5k16l &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/btwte2wv/checkpoints/epoch=06-val_loss=0.00.ckpt --id btwte2wv &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/lcs0r2r2/checkpoints/epoch=02-val_loss=0.00.ckpt --id lcs0r2r2 &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/pbb9qf7q/checkpoints/epoch=03-val_loss=0.00.ckpt --id pbb9qf7q &
wait
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/u3xgq33d/checkpoints/epoch=07-val_loss=0.00.ckpt --id u3xgq33d &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/0rilfhue/checkpoints/epoch=22-val_loss=0.00.ckpt --id 0rilfhue &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/2rj7ginm/checkpoints/epoch=03-val_loss=0.00.ckpt --id 2rj7ginm &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/6ttws8p2/checkpoints/epoch=21-val_loss=0.00.ckpt --id 6ttws8p2 &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/rj54z5s0/checkpoints/epoch=12-val_loss=0.00.ckpt --id rj54z5s0 &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/0jqx7512/checkpoints/epoch=03-val_loss=0.00.ckpt --id 0jqx7512 &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/3u4h1nmj/checkpoints/epoch=05-val_loss=0.00.ckpt --id 3u4h1nmj &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/44qbiqq6/checkpoints/epoch=07-val_loss=0.00.ckpt --id 44qbiqq6 &
wait
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/45awp2it/checkpoints/epoch=10-val_loss=0.00.ckpt --id 45awp2it &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/7cxess0a/checkpoints/epoch=06-val_loss=0.00.ckpt --id 7cxess0a &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/9kigfnra/checkpoints/epoch=08-val_loss=0.00.ckpt --id 9kigfnra &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/cms1q2ut/checkpoints/epoch=08-val_loss=0.00.ckpt --id cms1q2ut &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/ghbjya9c/checkpoints/epoch=08-val_loss=0.00.ckpt --id ghbjya9c &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/k4opv2rc/checkpoints/epoch=07-val_loss=0.00.ckpt --id k4opv2rc &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/obywu4hk/checkpoints/epoch=03-val_loss=0.00.ckpt --id obywu4hk &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/2roprevf/checkpoints/epoch=03-val_loss=0.00.ckpt --id 2roprevf &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/9qfdj6yk/checkpoints/epoch=02-val_loss=0.00.ckpt --id 9qfdj6yk &
wait
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/aj3vyl5a/checkpoints/epoch=01-val_loss=0.00.ckpt --id aj3vyl5a &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/en7uqoyt/checkpoints/epoch=04-val_loss=0.00.ckpt --id en7uqoyt &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/oj5evs1g/checkpoints/epoch=10-val_loss=0.00.ckpt --id oj5evs1g &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/s8onpwel/checkpoints/epoch=02-val_loss=0.00.ckpt --id s8onpwel &
uv run notebooks/predict.py --ckpt_path ./vmc24_mos/stnpzoj6/checkpoints/epoch=02-val_loss=0.00.ckpt --id stnpzoj6 &
wait