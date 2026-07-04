"""
Build-time weight prefetch (only runs when Dockerfile is built with --build-arg BAKE_WEIGHTS=1).
Downloads the main TRELLIS.2 weights into HF_HOME so cold starts don't pay the download cost.
Runs on CPU; if a GPU-only step fails on the build machine it is skipped safely.
"""
import os
import traceback

MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")

def main():
    from huggingface_hub import snapshot_download
    print(f"[weights] snapshot_download {MODEL_ID} -> {os.environ.get('HF_HOME')}", flush=True)
    snapshot_download(repo_id=MODEL_ID)

    # Best-effort: build the pipeline on CPU to cache auxiliary weights (bg-removal, etc.).
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
        print("[weights] auxiliary weights cached.", flush=True)
    except Exception:
        print("[weights] CPU pipeline build skipped (will fetch on first cold start):", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()
