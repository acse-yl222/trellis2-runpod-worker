"""
RunPod Serverless handler for TRELLIS.2 (image -> 3D GLB with PBR).

Signature is aligned to the official app.py `image_to_3d()` / `extract_glb()`:
  outputs, latents = pipeline.run(image, seed=..., preprocess_image=False,
      sparse_structure_sampler_params={...}, shape_slat_sampler_params={...},
      tex_slat_sampler_params={...}, pipeline_type=..., return_latent=True)
  shape_slat, tex_slat, res = latents
  mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
  glb  = o_voxel.postprocess.to_glb(...); glb.export(path, extension_webp=True)
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import io
import time
import base64
import tempfile
import traceback
from typing import Optional

import requests
import numpy as np
from PIL import Image

import torch
import runpod

# TRELLIS.2 imports (repo root is the cwd -> `trellis2` is importable)
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")
MAX_INLINE_MB = float(os.environ.get("MAX_INLINE_MB", "14"))
URL_EXPIRY_S = int(os.environ.get("URL_EXPIRY_S", str(7 * 24 * 3600)))

RESOLUTION_TO_PIPELINE = {
    "512": "512",
    "1024": "1024_cascade",
    "1536": "1536_cascade",
}

# Official defaults, taken verbatim from app.py's Advanced Settings.
DEFAULT_SS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
DEFAULT_SHAPE = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
DEFAULT_TEX = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        print(f"[init] loading pipeline {MODEL_ID} ...", flush=True)
        t0 = time.time()
        _pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
        _pipeline.cuda()
        print(f"[init] pipeline ready in {time.time() - t0:.1f}s", flush=True)
    return _pipeline


def _load_image(job_input: dict) -> Image.Image:
    """Load an RGBA PIL image from image_url or image_base64."""
    if job_input.get("image_url"):
        r = requests.get(job_input["image_url"], timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
    elif job_input.get("image_base64"):
        data = job_input["image_base64"]
        if "," in data and data.strip().startswith("data:"):
            data = data.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(data)))
    else:
        raise ValueError("Provide either 'image_url' or 'image_base64'.")
    return img.convert("RGBA")


def _merge(defaults: dict, override: Optional[dict]) -> dict:
    out = dict(defaults)
    if override:
        out.update({k: v for k, v in override.items() if v is not None})
    return out


def _maybe_upload(glb_path: str) -> dict:
    """Upload to S3/R2 if BUCKET_* env vars are set, else return base64."""
    size = os.path.getsize(glb_path)
    result = {"glb_size_bytes": size}

    if os.environ.get("BUCKET_ENDPOINT_URL"):
        from runpod.serverless.utils import rp_upload
        # rp_upload reads BUCKET_ENDPOINT_URL / BUCKET_ACCESS_KEY_ID / BUCKET_SECRET_ACCESS_KEY
        url = rp_upload.upload_file_to_bucket(
            file_name=os.path.basename(glb_path),
            file_location=glb_path,
        )
        result["glb_url"] = url
        return result

    mb = size / (1024 * 1024)
    if mb > MAX_INLINE_MB:
        raise ValueError(
            f"GLB is {mb:.1f}MB > MAX_INLINE_MB={MAX_INLINE_MB}. "
            f"Configure BUCKET_* env vars (S3/R2) to return a URL instead of base64."
        )
    with open(glb_path, "rb") as f:
        result["glb_base64"] = base64.b64encode(f.read()).decode()
    return result


def handler(job):
    try:
        job_input = job.get("input", {}) or {}
        pipeline = _load_pipeline()

        # --- inputs ---
        image = _load_image(job_input)
        seed = int(job_input.get("seed", 0))
        resolution = str(job_input.get("resolution", "1024"))
        if resolution not in RESOLUTION_TO_PIPELINE:
            raise ValueError(f"resolution must be one of {list(RESOLUTION_TO_PIPELINE)}")
        preprocess = bool(job_input.get("preprocess", True))
        decimation_target = int(job_input.get("decimation_target", 500000))
        texture_size = int(job_input.get("texture_size", 2048))

        ss = _merge(DEFAULT_SS, job_input.get("ss_sampler"))
        shape = _merge(DEFAULT_SHAPE, job_input.get("shape_sampler"))
        tex = _merge(DEFAULT_TEX, job_input.get("tex_sampler"))

        # --- preprocess (match app.py: preprocess separately, then run with preprocess_image=False) ---
        if preprocess:
            image = pipeline.preprocess_image(image)

        # --- generate ---
        t0 = time.time()
        _, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params=ss,
            shape_slat_sampler_params=shape,
            tex_slat_sampler_params=tex,
            pipeline_type=RESOLUTION_TO_PIPELINE[resolution],
            return_latent=True,
        )
        gen_s = time.time() - t0

        # --- decode + extract GLB (match app.py extract_glb) ---
        t1 = time.time()
        shape_slat, tex_slat, res = latents
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=False,
        )

        with tempfile.TemporaryDirectory() as td:
            glb_path = os.path.join(td, "output.glb")
            glb.export(glb_path, extension_webp=True)
            ext_s = time.time() - t1
            out = _maybe_upload(glb_path)

        torch.cuda.empty_cache()
        out["generation_time_s"] = round(gen_s, 2)
        out["extraction_time_s"] = round(ext_s, 2)
        out["resolution"] = resolution
        out["seed"] = seed
        return out

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
