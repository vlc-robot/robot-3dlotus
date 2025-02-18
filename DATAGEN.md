# Instructions on Dataset Generation

## Demonstration Generation and Preprocessing

We provide the scripts to generate training demonstrations below:

```bash
conda activate gembench

export sif_image=/scratch/shichen/singularity_images/nvcuda_v2.sif
export python_bin=$HOME/miniconda3/envs/gembench/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

seed=0
data_dir=data/gembench
img_size=256

microstep_data_dir=$data_dir/train_dataset/microsteps/seed${seed}
keystep_data_dir=$data_dir/train_dataset/keysteps_bbox/seed${seed}
keystep_pcd_dir=$data_dir/train_dataset/keysteps_bbox_pcd/seed${seed}

task=push_button
variation=3
num_episodes=100

# 1. Generate microsteps
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} preprocess/generate_dataset_microsteps.py \
    --microstep_data_dir ${microstep_data_dir} \
    --task ${task} --variation_id ${variation} --seed ${seed} \
    --image_size ${img_size} --renderer opengl \
    --episodes_per_task ${num_episodes} \    
    --live_demos

# 2. Extract keysteps of RGB-D observations
${python_bin} preprocess/generate_dataset_keysteps.py \
    --microstep_data_dir ${microstep_data_dir} \
    --keystep_data_dir ${keystep_data_dir} \
    --task ${task} --variation_id ${variation} \
    --image_size ${img_size} --save_masks

# 3. Convert to point clouds
${python_bin} preprocess/gen_simple_policy_data.py \
    --input_dir ${keystep_data_dir} \
    --output_dir ${keystep_pcd_dir} \
    --voxel_size 0.01 \
    --task ${task} --variation_id ${variation}
```

## Instruction Embedding Generation

```bash

# generate clip embeddings for all instructions
python preprocess/gen_instr_text_embeds.py \
    --input_file assets/taskvars_instructions_new.json \
    --output_dir your_output_dir \
    --model_name clip

# generate clip embeddings for action names
python preprocess/gen_action_text_embeds.py \
    --output_dir your_output_dir \
    --model_name clip
```