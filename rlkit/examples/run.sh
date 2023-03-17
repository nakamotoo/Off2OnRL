export CUDA_VISIBLE_DEVICES=0
export D4RL_SUPPRESS_IMPORT_ERROR=1
# export WANDB_DISABLED=True

# conda activate /home/user/.conda/envs/rlkit-original && cd /work/Off2OnRL/rlkit

export MUJOCO_PY_MUJOCO_PATH=/work/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MJKEY_PATH=/work/.mujoco/mjkey.txt
export PYTHONPATH=:/work/Off2OnRL/rlkit
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib
export LD_LIBRARY_PATH=/home/user/.conda/envs/odt/lib:$LD_LIBRARY_PATH

env=antmaze-medium-diverse-v2
# env=antmaze-medium-play-v2
# env=antmaze-large-play-v2
# env=antmaze-large-diverse-v2
# 1 3
# 2 4
# for seed in 1 3
for seed in 100
do
python examples/ours_cql.py \
--env_id $env \
--seed $seed \
--project off2on-antmaze \
--ensemble_size=1 \
--critic_num_hidden_layers=4 \
--cql_with_lagrange=0 \
--cql_alpha_weight=1.0
done
