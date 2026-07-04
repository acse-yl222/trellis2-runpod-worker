# TRELLIS.2 -> RunPod Serverless worker
# Base: CUDA 12.4 devel (nvcc needed to compile CuMesh / FlexGEMM / nvdiffrec at build time)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cover Ampere(8.0/8.6), Ada(8.9), Hopper(9.0). NOT Blackwell -- torch2.6/cu124 has no sm_120.
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    # Default HF cache lives in the image. Override to /runpod-volume/hf-cache when using a Network Volume.
    HF_HOME=/app/hf-cache \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential ninja-build \
        python3.10 python3.10-dev python3-pip \
        libjpeg-dev libgl1 libglib2.0-0 libegl1 libgles2 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- PyTorch (must match cu124) ----
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# ---- Clone TRELLIS.2 (provides the `trellis2` package + the o-voxel source) ----
# Pin a commit for reproducibility; bump when you want to track upstream.
ARG TRELLIS_REF=main
RUN git clone https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2 \
    && cd /app/TRELLIS.2 && git checkout ${TRELLIS_REF}

WORKDIR /app/TRELLIS.2

# ---- Basic python deps (mirrors setup.sh --basic) ----
RUN pip install \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
        trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
        kornia timm \
    && pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# pillow-simd is faster but often fails to build; fall back to plain pillow.
RUN pip install pillow-simd || pip install pillow

# ---- flash-attn (install the exact prebuilt wheel -> NO source compile) ----
# PyPI has no matching wheel, so pip would compile from source and blow past the
# builder's CPU time limit. Pin the official cu12/torch2.6/py310/abiFALSE wheel.
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ---- CUDA extensions (compiled here on RunPod's amd64 builder) ----
RUN mkdir -p /tmp/extensions \
    && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && pip install /tmp/extensions/nvdiffrast --no-build-isolation

RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec \
    && pip install /tmp/extensions/nvdiffrec --no-build-isolation

RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh \
    && pip install /tmp/extensions/CuMesh --no-build-isolation

RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM \
    && pip install /tmp/extensions/FlexGEMM --no-build-isolation

# ---- o-voxel (lives inside the TRELLIS.2 repo) ----
RUN pip install ./o-voxel --no-build-isolation

# ---- RunPod SDK + S3 upload deps ----
RUN pip install runpod boto3 requests

# ---- Worker code ----
COPY handler.py /app/TRELLIS.2/handler.py
COPY download_weights.py /app/TRELLIS.2/download_weights.py

# ---- Optional: bake model weights into the image ----
# Default 0 = do NOT bake (smaller/faster build; pair with a Network Volume for HF_HOME).
# Set --build-arg BAKE_WEIGHTS=1 to snapshot weights at build time (image ~25-35GB).
ARG BAKE_WEIGHTS=0
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python download_weights.py; fi

CMD ["python", "-u", "/app/TRELLIS.2/handler.py"]
