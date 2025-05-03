FROM ubuntu:24.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive
# Set timezone
ENV TZ=Europe/Paris

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    ffmpeg \
    libsm6 \
    libxext6

# Install CUDA toolkit and drivers and cudnn
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && apt install ./cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt update \
    && apt install -y cuda-toolkit libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Set up the CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# Verify CUDA installation
RUN nvcc --version

# Install Miniconda
RUN mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh \
    && /opt/miniconda3/bin/conda init bash

# Set PATH environment variable after installing Miniconda
ENV PATH="/opt/miniconda3/bin:${PATH}"

# Create and activate conda environment
RUN conda create -y -n gembench python=3.10

RUN conda run -n gembench --no-capture-output pip --default-timeout=2400 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Clone and install robot-3dlotus
RUN mkdir /opt/codes \
    && cd /opt/codes \
    && git clone https://github.com/vlc-robot/robot-3dlotus.git \
    && cd robot-3dlotus \
    && conda run -n gembench --no-capture-output pip install -r requirements.txt \
    && conda run -n gembench --no-capture-output pip install -e .

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
RUN conda run -n gembench --no-capture-output FORCE_CUDA=1 pip install torch-scatter==2.1.2

# Install chamferdist
RUN mkdir /opt/codes/robot-3dlotus/dependencies \
    && cd /opt/codes/robot-3dlotus/dependencies \
    && git clone https://github.com/cshizhe/chamferdist.git \
    && cd chamferdist \
    && conda run -n gembench --no-capture-output FORCE_CUDA=1 python setup.py install

# Install Pointnet2_PyTorch
RUN cd /opt/codes/robot-3dlotus/dependencies \
    && git clone https://github.com/cshizhe/Pointnet2_PyTorch.git \
    && cd Pointnet2_PyTorch/pointnet2_ops_lib \
    && conda run -n gembench --no-capture-output FORCE_CUDA=1 python setup.py install

# Install llama
RUN cd /opt/codes/robot-3dlotus/dependencies \
    && git clone https://github.com/cshizhe/llama3.git \
    && cd llama3 \
    && conda run -n gembench --no-capture-output pip install -e .

# Download models
# 3D-Lotus and 3D-Lotus++
RUN mkdir -p /opt/codes/robot-3dlotus/data/experiments/3dlotus \
    && conda run -n gembench --no-capture-output \
       huggingface-cli download --repo-type model \
       --local-dir-use-symlinks False \
       --local-dir /opt/codes/robot-3dlotus/data/experiments/3dlotus \
       rjgpinel/3dlotus --include GEMBench/ \
    && conda run -n gembench --no-capture-output \
        huggingface-cli download --repo-type model \
        --local-dir-use-symlinks False \
        --local-dir /opt/codes/robot-3dlotus/data/experiments/3dlotusplus \
        rjgpinel/3dlotusplus --include GEMBench/

# CLIP ViT for text encoding
RUN conda run -n gembench --no-capture-output \
    huggingface-cli download --repo-type model \
    openai/clip-vit-base-patch32 \
    --exclude tf_model.h5 flax_model.msgpack

# SAM for segmentation
RUN conda run -n gembench --no-capture-output \
    huggingface-cli download --repo-type model \
    facebook/sam-vit-huge \
    --exclude pytorch_model.bin tf_model.h5

# Owlv2 for open-vocabulary object detection
RUN conda run -n gembench --no-capture-output \
    huggingface-cli download --repo-type model \
    google/owlv2-large-patch14-ensemble \
    --exclude pytorch_model.bin

# LLama3 for task planning: please replace `Your Token ID` with your actual id
RUN conda run -n gembench --no-capture-output \
    huggingface-cli login --token `Your Token ID` \
    && conda run -n gembench --no-capture-output huggingface-cli download --repo-type model \
       --local-dir-use-symlinks False \
       --local-dir /opt/codes/robot-3dlotus/data/pretrained/meta-llama/Meta-Llama-3-8B-Instruct \
       meta-llama/Meta-Llama-3-8B-Instruct \
       --include original/

# Bert model for text similarity measure
RUN conda run -n gembench --no-capture-output \
    huggingface-cli download --repo-type model \
    sentence-transformers/all-MiniLM-L6-v2 \
    --exclude openvino onnx 1_Pooling

# Clean up cache
RUN conda run -n gembench --no-capture-output pip cache purge \
    && conda clean --all -y

# Set working directory
WORKDIR /opt/codes/robot-3dlotus

# Default command when container starts
CMD ["/bin/bash"]
