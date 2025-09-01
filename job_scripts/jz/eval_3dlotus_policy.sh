#!/bin/bash
#SBATCH --job-name=eval_policy
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
#SBATCH --time=2:00:00
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

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

sif_image=$SINGULARITY_ALLOWED_DIR/nvcuda_v2.sif
export python_bin=$HOME/miniconda3/envs/gondola/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

expr_dir=data/experiments/uarm/gembench/exp008
ckpt_step=$1

# validation
singularity exec --bind $HOME:$HOME,$WORK:$WORK,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_simple_policy_server.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 4 \
    --taskvar_file assets/taskvars_train.json \
    --seed 100 --num_demos 20 \
    --microstep_data_dir data/gembench/val_dataset/microsteps/seed100 \
    --cam_ids 0 1 2

