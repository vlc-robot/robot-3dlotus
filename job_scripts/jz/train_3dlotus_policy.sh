#!/bin/bash
#SBATCH --job-name=train_policy
#SBATCH -A qji@v100
#SBATCH -C v100-32g
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


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_TASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

ulimit -n 2048


rot_type=euler_disc
npoints=4096
pos_bin_size=15

output_dir=data/experiments/uarm/gembench/exp010
#trn_data_dir=data/gembench/train_dataset/keysteps_bbox_pcd/seed0/voxel1cm
trn_data_dir=data/gembench/train_dataset/keysteps_bbox_pcd_cam12/seed0
val_data_dir=data/gembench/val_dataset/keysteps_bbox_pcd/seed100/voxel1cm

srun python genrobo3d/train/train_simple_policy.py \
    --exp-config genrobo3d/configs/rlbench/simple_policy_ptv3.yaml \
    output_dir ${output_dir} \
    TRAIN.num_epochs null TRAIN.num_train_steps 150000 \
    TRAIN.log_steps 1000 TRAIN.save_steps 10000 TRAIN.val_steps 10000 \
    TRAIN.train_batch_size 8 TRAIN.val_batch_size 8 \
    VAL_DATASET.use_val True \
    TRAIN_DATASET.rm_robot box_keep_gripper VAL_DATASET.rm_robot box_keep_gripper \
    TRAIN_DATASET.num_points ${npoints} VAL_DATASET.num_points ${npoints} \
    TRAIN_DATASET.all_step_in_batch True VAL_DATASET.all_step_in_batch True \
    TRAIN_DATASET.instr_embed_type all VAL_DATASET.instr_embed_type all \
    TRAIN_DATASET.xyz_shift center VAL_DATASET.xyz_shift center \
    TRAIN_DATASET.xyz_norm False VAL_DATASET.xyz_norm False \
    TRAIN_DATASET.rot_type ${rot_type} VAL_DATASET.rot_type ${rot_type} \
    TRAIN_DATASET.taskvar_file assets/taskvars_train.json VAL_DATASET.taskvar_file assets/taskvars_train.json \
    TRAIN_DATASET.data_dir ${trn_data_dir} \
    VAL_DATASET.data_dir ${val_data_dir} \
    TRAIN_DATASET.include_last_step False VAL_DATASET.include_last_step False \
    TRAIN_DATASET.use_height True VAL_DATASET.use_height True \
    TRAIN_DATASET.augment_pc True VAL_DATASET.augment_pc False \
    TRAIN_DATASET.aug_max_rot 180 \
    TRAIN_DATASET.rm_pc_outliers False VAL_DATASET.rm_pc_outliers False \
    MODEL.ptv3_config.drop_path 0.0 MODEL.ptv3_config.attn_drop 0.1 MODEL.ptv3_config.proj_drop 0.1 \
    MODEL.action_config.dropout 0.2 \
    MODEL.action_config.voxel_size 0.01 \
    MODEL.action_config.reduce max \
    MODEL.action_config.dim_actions 7 MODEL.action_config.rot_pred_type ${rot_type} \
    MODEL.action_config.pos_heatmap_temp 0.1 \
    MODEL.ptv3_config.in_channels 7 \
    MODEL.ptv3_config.pdnorm_only_decoder False \
    MODEL.ptv3_config.qk_norm True \
    MODEL.ptv3_config.scaled_cosine_attn False MODEL.ptv3_config.enable_flash False \
    MODEL.action_config.max_steps 30 \
    MODEL.ptv3_config.enc_depths "[1, 1, 1, 1, 1]" \
    MODEL.ptv3_config.dec_depths "[1, 1, 1, 1]" \
    MODEL.ptv3_config.enc_channels "[64, 128, 256, 512, 768]" \
    MODEL.ptv3_config.dec_channels "[128, 128, 256, 512]" \
    MODEL.action_config.use_step_id False \
    MODEL.action_config.use_ee_pose False \
    MODEL.loss_config.pos_weight 1 MODEL.loss_config.rot_weight 1 \
    MODEL.action_config.pos_pred_type heatmap_disc \
    TRAIN_DATASET.pos_type disc VAL_DATASET.pos_type disc \
    TRAIN_DATASET.pos_heatmap_type dist VAL_DATASET.pos_heatmap_type dist \
    TRAIN_DATASET.pos_bins ${pos_bin_size} VAL_DATASET.pos_bins ${pos_bin_size} \
    MODEL.action_config.pos_bins ${pos_bin_size} \
    TRAIN_DATASET.pos_heatmap_no_robot True VAL_DATASET.pos_heatmap_no_robot True \
    MODEL.model_class SimplePolicyPTV3CA \
    MODEL.ptv3_config.pdnorm_bn False MODEL.ptv3_config.pdnorm_ln False \
    MODEL.ptv3_config.pdnorm_adaptive False \
    TRAIN.resume_training True
