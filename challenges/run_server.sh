#! /bin/bash

cd $HOME/codes/robot-3dlotus

model_name=$1   # 3dlotusplus, 3dlotus
port=$2         # 13000

conda run -n gembench --no-capture-output python challenges/server.py \
    --port ${port} --model ${model_name}