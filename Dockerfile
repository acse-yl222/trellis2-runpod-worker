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
    # setuptools>=64 is REQUIRED so pip honors PEP 621 pyproject.toml metadata.
    # nvdiffrast v0.4.0 declares its package list ONLY in pyproject.toml; the stock
    # Ubuntu setuptools (~59) ignores it and installs the CUDA ext without the
    # `nvdiffrast` python package -> runtime "No module named 'nvdiffrast'".
    && python -m pip install --upgrade "pip" "setuptools>=70" "wheel" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- PyTorch (must match cu124) ----
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# ---- Clone TRELLIS.2 (provides the `trellis2` package + the o-voxel source) ----
# Pin a commit for reproducibility; bump when you want to track upstream.
ARG TRELLIS_REF=main
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2 \
    && cd /app/TRELLIS.2 && git checkout ${TRELLIS_REF} \
    && git submodule update --init --recursive

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
# Verify each import in the SAME layer it is installed, so a silently-broken install
# fails the BUILD instead of shipping a worker that dies on the first job. Editing
# these RUN lines also busts any poisoned layer cache for nvdiffrast and everything
# after it -- the "No module named 'nvdiffrast'" job crash came from a stale cached layer.
RUN mkdir -p /tmp/extensions \
    && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && pip install /tmp/extensions/nvdiffrast --no-build-isolation \
    && python -c "import nvdiffrast.torch; print('[verify] nvdiffrast OK')"

RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec \
    && pip install /tmp/extensions/nvdiffrec --no-build-isolation

RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh \
    && pip install /tmp/extensions/CuMesh --no-build-isolation

RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM \
    && pip install /tmp/extensions/FlexGEMM --no-build-isolation

# ---- o-voxel (lives inside the TRELLIS.2 repo) ----
# o-voxel compiles its CUDA ext here. We do NOT `import o_voxel` at build: importing it
# loads o_voxel._C which dlopens libcuda, and the BUILD host has no GPU/driver, so the
# import would fail spuriously. Instead confirm the package is installed via find_spec
# (does not execute o_voxel/__init__). The real import runs at runtime on a GPU, and the
# handler imports lazily so any genuine issue surfaces as a job error, not a crash.
RUN pip install ./o-voxel --no-build-isolation \
    && python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('o_voxel') else 1)" \
    && echo "[verify] o_voxel package installed (runtime import deferred to GPU)"

# ---- RunPod SDK + S3 upload deps ----
RUN pip install runpod boto3 requests

# ---- Pin transformers for DINOv3 compatibility (CRITICAL) ----
# TRELLIS.2's DinoV3FeatureExtractor.extract_features (trellis2/modules/image_feature_extractor.py)
# reaches into the DINOv3 model internals: it uses model.embeddings, model.rope_embeddings and
# iterates `model.layer` DIRECTLY. transformers 5.x refactored DINOv3ViTModel to nest the encoder
# under `.model` (the layers moved to model.model.layer), so the flat `.layer` attribute is gone:
#     'DINOv3ViTModel' object has no attribute 'layer'
# 4.56.0–4.57.1 keep the flat layout TRELLIS.2 was written against. Pin the last compatible (4.57.1).
# NOTE: the base `transformers` above is left unpinned only because this line overrides it; keep this
# RUN AFTER the CUDA-extension layers so re-pinning never busts those slow source compiles.
RUN pip install "transformers==4.57.1"

# ---- Pillow: force ONE self-consistent build for webp GLB export (CRITICAL) ----
# `pillow-simd || pillow` above + other deps pulling Pillow 11 leaves a MIXED PIL: an old
# WebPImagePlugin.py that references _webp.HAVE_WEBPANIM next to a Pillow-11 _webp C-ext that
# dropped it, so glb.export(extension_webp=True) dies with:
#     module 'PIL._webp' has no attribute 'HAVE_WEBPANIM'
# Wipe every Pillow variant and reinstall a single clean 10.4.0 (plugin + C-ext both expose
# HAVE_WEBPANIM). Late layer -> does not bust the CUDA-extension cache.
RUN pip uninstall -y Pillow Pillow-SIMD pillow-simd pillow 2>/dev/null; \
    pip install --force-reinstall --no-cache-dir "Pillow==10.4.0"

# ---- Worker code ----
COPY handler.py /app/TRELLIS.2/handler.py
COPY download_weights.py /app/TRELLIS.2/download_weights.py

# ---- Bake model weights into the image ----
# Default 1 = bake the 16GB TRELLIS.2-4B snapshot at build time so cold starts do NOT
# re-download it (image grows to ~30-35GB; build needs enough builder disk). The small
# gated DINOv3 + BiRefNet models are NOT baked here (DINOv3 needs a build-time HF token);
# they download once per cold worker via the endpoint's HF_TOKEN env var (~2GB, fast).
# Set --build-arg BAKE_WEIGHTS=0 to skip baking (smaller/faster build, but every cold
# start re-downloads the 16GB).
ARG BAKE_WEIGHTS=1
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python download_weights.py; fi

CMD ["python", "-u", "/app/TRELLIS.2/handler.py"]
