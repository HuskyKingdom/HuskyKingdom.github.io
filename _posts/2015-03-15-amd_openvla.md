---
layout: post
title: Running OpenVLA Evaluation (LIBERO) on AMD Servers
date: 2025-10-16 11:59:00-0400
description: This guide provides a standardized operating procedure (SOP) for running OpenVLA robotic manipulation evaluations (LIBERO Benchmark) on AMD GPU high-performance computing (HPC) clusters using Apptainer/Singularity and SLURM.
categories: tutorial
tags: amd, openvla, embodiedai
disqus_comments: true
related_posts: false
toc:
  sidebar: left
---


# 🚀 Running OpenVLA Evaluation (LIBERO) on AMD Servers

This guide provides a standardized operating procedure (SOP) for running OpenVLA robotic manipulation evaluations (LIBERO Benchmark) on AMD GPU high-performance computing (HPC) clusters using Apptainer/Singularity and SLURM.

---

## Step 1: Workspace Preparation

Clone the necessary repositories into your workspace root (e.g., `/work1/username/workspace`).

```bash
# 1. Clone OpenVLA (or your custom fork, e.g., openvla-oft-yhs)
git clone [https://github.com/moojink/openvla-oft.git](https://github.com/moojink/openvla-oft.git)

# 2. Clone LIBERO Benchmark
git clone [https://github.com/Lifelong-Robot-Learning/LIBERO.git](https://github.com/Lifelong-Robot-Learning/LIBERO.git)
```

---

## Step 2: Build the Bulletproof Apptainer Image

We consolidate all fragmented dependencies into a single Apptainer definition file to create a plug-and-play environment. 

### 1. Create `openvla_rocm.def`
Create this file in your root directory and paste the following content:

```apptainer
Bootstrap: docker
From: rocm/primus:v26.1

%environment
    export PATH="/opt/ompi/bin:/opt/ucx/bin:/opt/rocm/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/lib:/opt/rocm/lib:/opt/ompi/lib:/opt/ucx/lib:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs:$LD_LIBRARY_PATH"
    
    # CRITICAL: Headless rendering settings for LIBERO
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    export ROBOT_PLATFORM=LIBERO
    
    export NUMBA_DISABLE_JIT=1
    export TOKENIZERS_PARALLELISM=false

%post
    # 1. System dependencies & OSMesa rendering libraries
    apt-get update
    apt-get install -y  \
            librdmacm1 libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils rdma-core \
            git cmake gcc g++ make automake autoconf libtool \
            pkg-config numactl libnuma-dev librdmacm-dev \
            flex bison wget curl libdrm-dev libibumad-dev \
            pciutils ethtool libpci-dev \
            libegl1-mesa-dev libgles2-mesa-dev \
            libosmesa6 libosmesa6-dev libegl1-mesa libgl1-mesa-glx libgles2-mesa

    # Fix potential ROCm RDMA driver conflicts
    mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.inbox 2>/dev/null || true
    ldconfig

    # 2. Basic tools (replace GUI-based OpenCV with headless)
    pip3 uninstall -y opencv-python || true
    pip3 install opencv-python-headless==4.11.0.86 packaging ninja torchvision torchaudio

    # 3. Core LLM/VLM dependencies
    pip3 install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.16
    pip3 install draccus==0.8.0 "sentencepiece>=0.1.99" json-numpy tensorflow-cpu==2.15.1

    # 4. Install LIBERO Environment
    cd /
    git clone [https://github.com/Lifelong-Robot-Learning/LIBERO.git](https://github.com/Lifelong-Robot-Learning/LIBERO.git)
    touch LIBERO/libero/__init__.py
    pip3 install -e LIBERO
    pip3 install "imageio[ffmpeg]" robosuite==1.4.1 bddl easydict cloudpickle gym

    # Pre-create LIBERO config to avoid runtime CLI prompts
    mkdir -p /root/.libero
    python3 -c "
import os, yaml
root = '/LIBERO/libero/libero'
config = {
    'benchmark_root': root,
    'bddl_files': os.path.join(root, 'bddl_files'),
    'init_states': os.path.join(root, 'init_files'),
    'datasets': os.path.join(root, '../datasets'),
    'assets': os.path.join(root, 'assets'),
}
with open('/root/.libero/config.yaml', 'w') as f:
    yaml.dump(config, f)
"

    # 5. Install OpenVLA
    cd /
    git clone [https://github.com/moojink/openvla-oft.git](https://github.com/moojink/openvla-oft.git)
    cd openvla-oft
    pip3 install -e . --no-deps
    pip3 install -r experiments/robot/libero/libero_requirements.txt 

    # 6. Ultimate Dependency Fix: Overwrite conflicting packages
    echo "=== Force installing robust versions ==="
    pip3 install -U transformers==4.44.0 peft==0.17.0 diffusers==0.30.2
    pip3 install tensorflow-datasets tensorflow-graphics jsonlines
    
    # Strictly lock Numpy and ml_dtypes to prevent C-API crashes; Install authentic dlimp from GitHub
    pip3 install --no-deps --force-reinstall ml_dtypes==0.3.1 numpy==1.24.4 git+[https://github.com/kvablack/dlimp.git](https://github.com/kvablack/dlimp.git)

%help
    OpenVLA-OFT environment fixed for AMD ROCm & LIBERO.
```

### 2. Build the Image
**CRITICAL:** HPC login nodes usually have very restricted `/tmp` storage. You must redirect Apptainer's cache and temporary directories to your high-capacity storage (e.g., `/work1/`) before building, otherwise the build will crash with a `stream error`.

```bash
# Redirect temp directories to high-capacity storage
export APPTAINER_TMPDIR=/path/to/your/large/storage/apptainer_tmp
export APPTAINER_CACHEDIR=/path/to/your/large/storage/apptainer_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# Clean old cache and build
apptainer cache clean --force
apptainer build openvla_rocm.sif openvla_rocm.def
```

---

## Step 3: Source Code Patches

Before running the evaluation, you must apply two minor patches to the Python script located at `openvla-oft/experiments/robot/libero/run_libero_eval.py`.

### Patch 1: Bypass PyTorch 2.6 `weights_only` Restriction
PyTorch 2.6 enforces `weights_only=True` by default, which blocks the loading of LIBERO's initial state files (`.init`). 
Open `run_libero_eval.py` and inject the following patch immediately after `import torch` at the top of the file:

```python
import torch

# ================= PATCH START =================
# Bypass PyTorch 2.6 weights_only restriction to load LIBERO states
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ================= PATCH END =================
```

### Patch 2: Remove Redundant Arguments (For custom forks only)
If you are using a custom/modified branch (like `openvla-oft-yhs`) and encounter a `TypeError: get_action() got an unexpected keyword argument 'h_head'`, locate the `get_action()` call inside the episode loop (around line 360) and remove or comment out the `h_head=...` argument.

---

## Step 4: SLURM Submission Script (`slurm.sh`)

Create your SLURM submission script (`slurm.sh`). Pay special attention to the `--bind` mounts to ensure the container can access your datasets located outside the repository folder.

```bash
#!/bin/bash
#SBATCH --job-name=eval-openvla-libero
#SBATCH --partition=mi3508x       # Replace with your specific partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # Evaluation only requires 1 GPU
#SBATCH --cpus-per-task=60 
#SBATCH --time=5:00:00
#SBATCH --output=slurm/logs/eval_openvla_libero_%j.out
#SBATCH --error=slurm/logs/eval_openvla_libero_%j.err

set -ex

# 1. Paths (Adjust REPO_ROOT to your workspace path)
REPO_ROOT="${SLURM_SUBMIT_DIR}"
APPTAINER_IMAGE="$REPO_ROOT/openvla_rocm.sif"

# 2. Evaluation Config
PRETRAINED_CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial"
EVAL_SCRIPT="openvla-oft/experiments/robot/libero/run_libero_eval.py"
DATASET_NAME="libero_spatial"

# 3. Runtime Environment (Headless Rendering & GPU Isolation)
export ROBOT_PLATFORM="LIBERO"
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export EGL_PLATFORM="surfaceless"
export HIP_VISIBLE_DEVICES="0"

# Redirect HuggingFace & Triton caches into the workspace
export NUMBA_CACHE_DIR="$REPO_ROOT/.cache/numba"
export HF_CACHE_DIR="$REPO_ROOT/.cache/huggingface"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache/triton"
mkdir -p "$NUMBA_CACHE_DIR" "$HF_CACHE_DIR" "$TRITON_CACHE_DIR"

export LD_LIBRARY_PATH="/usr/local/lib:/opt/rocm/lib:/opt/ompi/lib:/opt/ucx/lib:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs"

# 4. Command to run inside the container
INNER_CMD="set -ex; \
cd \"$REPO_ROOT\"; \
python -u \"$EVAL_SCRIPT\" \
  --pretrained_checkpoint \"$PRETRAINED_CHECKPOINT\" \
  --task_suite_name \"$DATASET_NAME\""

# 5. Run Apptainer Container
# CRITICAL: Ensure you --bind the parent directory where your LIBERO benchmarks/datasets are located (e.g., /work1/username/)
apptainer run \
  --cleanenv \
  --writable-tmpfs \
  --env "ROBOT_PLATFORM=$ROBOT_PLATFORM" \
  --env "MUJOCO_GL=$MUJOCO_GL" \
  --env "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM" \
  --env "EGL_PLATFORM=$EGL_PLATFORM" \
  --env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" \
  --env "NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR" \
  --env "TRANSFORMERS_CACHE=$HF_CACHE_DIR" \
  --env "HF_HOME=$HF_CACHE_DIR" \
  --env "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" \
  --env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
  --bind "$REPO_ROOT:$REPO_ROOT" \
  --bind "/work1/chunyilee/yuhang:/work1/chunyilee/yuhang" \
  "$APPTAINER_IMAGE" \
  bash -c "$INNER_CMD"
```

---

## Step 5: Submission & Monitoring

Create the logs directory and submit the job:

```bash
mkdir -p slurm/logs
sbatch slurm.sh
```

Monitor the execution logs:
```bash
tail -f slurm/logs/eval_openvla_libero_<JOB_ID>.out
```
