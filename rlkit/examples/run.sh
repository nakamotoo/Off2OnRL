export CUDA_VISIBLE_DEVICES=7
export D4RL_SUPPRESS_IMPORT_ERROR=1
# export WANDB_DISABLED=True


env=antmaze-medium-diverse-v2
# env=antmaze-medium-play-v2
# env=antmaze-large-play-v2
# env=antmaze-large-diverse-v2
# 1 3
# 2 4
# for seed in 1 3
for seed in 2 4
do
python examples/ours_cql.py \
--env_id $env \
--seed $seed \
--project off2on-antmaze
done
