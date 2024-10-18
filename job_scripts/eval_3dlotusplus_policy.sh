#!/bin/bash
#SBATCH --job-name=eval_policy
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out
#SBATCH -p willow 
#SBATCH -A willow

set -x
set -e

module purge
pwd; hostname; date

cd $HOME/Projects/robot-3dlotus

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gembench


export sif_image=/scratch/ppacaud/singularity_images/nvcuda_v2.sif
export python_bin=$HOME/miniconda3/envs/gembench/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

expr_dir=data/experiments/gembench/3dlotusplus/v1
ckpt_step=140000

# validation: with groundtruth task planner and groundtruth object grounding
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_robot_pipeline_server.py \
    --full_gt \
    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline_gt.yaml \
    --mp_expr_dir ${expr_dir} \
    --mp_ckpt_step ${ckpt_step} \
    --num_workers 4 \
    --taskvar_file assets/taskvars_train_debug.json \
    --gt_og_label_file assets/taskvars_target_label_zrange.json \
    --seed 100 --num_demos 20 \
    --microstep_data_dir data/gembench/val_dataset/microsteps/seed100 \
    --pc_label_type coarse --run_action_step 1

# test: with groundtruth task planner and groundtruth object grounding
#for seed in {200..600..100}
#do
#for split in train test_l2 test_l3 test_l4
#do
#singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_robot_pipeline_server.py \
#    --full_gt \
#    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline_gt.yaml \
#    --mp_expr_dir ${expr_dir} \
#    --mp_ckpt_step ${ckpt_step} \
#    --num_workers 4 \
#    --taskvar_file assets/taskvars_${split}.json \
#    --gt_og_label_file assets/taskvars_target_label_zrange.json \
#    --seed ${seed} --num_demos 20 \
#    --microstep_data_dir data/gembench/test_dataset/microsteps/seed${seed} \
#    --pc_label_type coarse --run_action_step 1
#done
#done
#
## test: with groundtruth task planner and automatic object grounding
#for seed in {200..600..100}
#do
#for split in train test_l2 test_l3 test_l4
#do
#singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_robot_pipeline_server.py \
#    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline.yaml \
#    --mp_expr_dir ${expr_dir} \
#    --mp_ckpt_step ${ckpt_step} \
#    --num_workers 4 \
#    --taskvar_file assets/taskvars_${split}.json \
#    --seed ${seed} --num_demos 20 \
#    --microstep_data_dir data/gembench/test_dataset/microsteps/seed${seed} \
#    --pc_label_type coarse --run_action_step 5
#done
#done
#
## test: full automatic
#for seed in {200..600..100}
#do
#for split in train test_l2 test_l3 test_l4
#do
#singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_robot_pipeline_server.py \
#    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline.yaml \
#    --mp_expr_dir ${expr_dir} \
#    --mp_ckpt_step ${ckpt_step} \
#    --num_workers 4 \
#    --taskvar_file assets/taskvars_${split}.json \
#    --seed ${seed} --num_demos 20 \
#    --microstep_data_dir data/gembench/test_dataset/microsteps/seed${seed} \
#    --pc_label_type coarse --run_action_step 5 \
#    --no_gt_llm --llm_master_port 15322
#    # --llm_cache_file data/experiments/gembench/llm_planner/llama3_8b_noobj/${split}.jsonl
#done
#done
