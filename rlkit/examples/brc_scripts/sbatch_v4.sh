#!/bin/bash
#SBATCH --job-name=lp4-5
#SBATCH --open-mode=append
#SBATCH --output=logs/out/%x_%j.txt
#SBATCH --error=logs/err/%x_%j.txt
#SBATCH --time=240:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TITAN:1
#SBATCH --account=co_rail
#SBATCH --partition=savio3_gpu
#SBATCH --qos=rail_gpu3_normal

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))

BETAS="1;2;"

arrBETAS=(${BETAS//;/ })

BETA=${arrBETAS[$TASK_ID]}

module load gnu-parallel

export PROJECT_DIR=/global/home/users/$USER/Off2OnRL/rlkit
export LOG_DIR=/global/scratch/users/$USER/Off2OnRL
export PROJECT_NAME="odt-antmaze"

run_singularity ()
{
singularity exec --nv --writable-tmpfs -B /usr/lib64 -B /var/lib/dcv-gl --overlay /global/scratch/users/nakamoto/singularity/overlay-50G-10M.ext3:ro /global/scratch/users/nakamoto/singularity/cuda11.5-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
    source ~/.bashrc
    conda activate odt
    cd $PROJECT_DIR
    python examples/ours_cql.py \
    --env_id $2 \
    --seed $1 \
    --project off2on-antmaze-maxqbackup-sweep \
    --ensemble_size=5 \
    --critic_num_hidden_layers=$4 \
    --cql_with_lagrange=1 \
    --cql_alpha_weight=5.0 \
    --target_action_gap=$3 \
"
}

export -f run_singularity
parallel --delay 20 --linebuffer -j 1 run_singularity $BETA {} \
    ::: antmaze-large-play-v2 \
    ::: 5 \
    ::: 4