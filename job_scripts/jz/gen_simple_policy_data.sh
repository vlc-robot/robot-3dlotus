#!/bin/bash
#SBATCH --job-name=gen_policy_data
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


input_dir=data/gembench/train_dataset/keysteps_bbox/seed0
output_dir=data/gembench/train_dataset/keysteps_bbox_pcd_cam12/seed0

${python_bin} preprocess/gen_simple_policy_data.py \
    --input_dir ${input_dir} \
    --output_dir ${output_dir} \
    --cam_ids 1 2
    
