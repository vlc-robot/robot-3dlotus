#!/bin/bash
#SBATCH --job-name=regen_microstep
#SBATCH -A qji@v100
##SBATCH -C v100-16g
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3 # t3 t4(100h)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
##SBATCH --mem=40G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out


module purge
module load singularity
module load cuda/12.1.0
module load gcc/11.3.1

set -x
set -e

pwd; hostname; date

cd $HOME/codes/robot-3dlotus

source $HOME/.bashrc
conda activate gondola

sif_image=$SINGULARITY_ALLOWED_DIR/nvcuda_v2.sif
export python_bin=$HOME/miniconda3/envs/gondola/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

microstep_data_dir=$SCRATCH/datasets/RLBench/gembench/train_dataset/microsteps/seed0
prev_state_dir=$SCRATCH/datasets/RLBench/gembench/train_dataset/microsteps_released/seed0
seed=0
img_size=256
num_episodes=100

task=$1 #push_button
variation=$2 #3

singularity exec --bind $HOME:$HOME,$WORK:$WORK,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} preprocess/generate_dataset_microsteps.py \
    --microstep_data_dir ${microstep_data_dir} \
    --task ${task} --variation_id ${variation} --seed ${seed} \
    --image_size ${img_size} --renderer opengl \
    --episodes_per_task ${num_episodes} \
    --prev_state_dir ${prev_state_dir}
    # --live_demos
    #--prev_state_dir ${prev_state_dir}
    
