# ============================================================================
# Position Encoding Experiments - Docker Image
# ============================================================================
#
# Build:
#   docker build -t pos_encoding_exp .
#
# Run experiments:
#   docker run --gpus all -v ${PWD}:/workspace pos_encoding_exp python experiment_argmax.py --bugfix
#   docker run --gpus all -v ${PWD}:/workspace pos_encoding_exp python experiment_mlm_nlp.py --bugfix
#   docker run --gpus all -v ${PWD}:/workspace pos_encoding_exp python experiment_mlm_protein.py --bugfix
#
# Full sweeps:
#   docker run --gpus all --shm-size=64g -v ${PWD}:/workspace pos_encoding_exp python experiment_argmax.py
#   docker run --gpus all --shm-size=64g -v ${PWD}:/workspace pos_encoding_exp python experiment_mlm_nlp.py
#   docker run --gpus all --shm-size=64g -v ${PWD}:/workspace pos_encoding_exp python experiment_mlm_protein.py
#
# Interactive shell:
#   docker run --gpus all -it -v ${PWD}:/workspace pos_encoding_exp bash
#
# ============================================================================

# 1. CUDA / cuDNN base
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 2. System prerequisites + Python 3.12
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.12.7 \
    PATH=/usr/local/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates ninja-build \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}* && \
    ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12    /usr/local/bin/pip

# 3. Project code location (inside image)
WORKDIR /app

# 4. Install dependencies (cached layer)
COPY requirements.txt .

RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt -U && \
    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128 -U && \
    pip install numpy==1.26.4

# 5. Copy project source
COPY . .

# 6. Working directory = bind-mount target
WORKDIR /workspace

# 7. Environment: single host volume for all artefacts & caches
ENV PROJECT_ROOT=/workspace \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache

RUN mkdir -p \
      /workspace/.cache/huggingface \
      /workspace/.cache/torch \
      /workspace/outputs \
      /workspace/figures

VOLUME ["/workspace"]

# 8. Default: interactive shell
CMD ["bash"]
