#!/usr/bin/env python3
"""
Call a deployed TRELLIS.2 RunPod endpoint.

Examples:
  export RUNPOD_API_KEY=rpa_xxx
  python client.py --endpoint-id <ID> --image my_object.png --resolution 1024 -o out.glb
  python client.py --endpoint-id <ID> --image-url https://... --resolution 1536 --mode async -o out.glb
"""
import os
import sys
import time
import json
import base64
import argparse
import urllib.request


def _post(url, api_key, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1000) as r:
        return json.loads(r.read().decode())


def _get(url, api_key):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def save_output(out, out_path):
    if out.get("error"):
        print("ERROR from worker:", out["error"])
        if out.get("trace"):
            print(out["trace"])
        sys.exit(1)
    if out.get("glb_url"):
        print("GLB URL:", out["glb_url"])
        try:
            urllib.request.urlretrieve(out["glb_url"], out_path)
            print("Saved:", out_path)
        except Exception as e:
            print("(could not auto-download, open the URL manually)", e)
    elif out.get("glb_base64"):
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(out["glb_base64"]))
        print("Saved:", out_path)
    else:
        print("Unexpected output:", json.dumps(out, indent=2)[:2000])
        return
    for k in ("generation_time_s", "extraction_time_s", "glb_size_bytes"):
        if k in out:
            print(f"  {k}: {out[k]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-id", required=True)
    p.add_argument("--image", help="local image path")
    p.add_argument("--image-url", help="public image URL")
    p.add_argument("--resolution", default="1024", choices=["512", "1024", "1536"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--decimation-target", type=int, default=500000)
    p.add_argument("--texture-size", type=int, default=2048, choices=[1024, 2048, 4096])
    p.add_argument("--mode", default="sync", choices=["sync", "async"])
    p.add_argument("-o", "--output", default="out.glb")
    args = p.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("Set RUNPOD_API_KEY env var (console -> Settings -> API Keys).")

    inp = {
        "resolution": args.resolution,
        "seed": args.seed,
        "preprocess": not args.no_preprocess,
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
    }
    if args.image_url:
        inp["image_url"] = args.image_url
    elif args.image:
        with open(args.image, "rb") as f:
            inp["image_base64"] = base64.b64encode(f.read()).decode()
    else:
        sys.exit("Provide --image or --image-url")

    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    payload = {"input": inp}

    if args.mode == "sync":
        print("Submitting (runsync)...")
        out = _post(f"{base}/runsync", api_key, payload)
        save_output(out.get("output", out), args.output)
    else:
        print("Submitting (run)...")
        job = _post(f"{base}/run", api_key, payload)
        job_id = job["id"]
        print("job id:", job_id)
        while True:
            time.sleep(3)
            st = _get(f"{base}/status/{job_id}", api_key)
            status = st.get("status")
            print("  status:", status)
            if status == "COMPLETED":
                save_output(st.get("output", {}), args.output)
                break
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                print(json.dumps(st, indent=2)[:2000])
                sys.exit(1)


if __name__ == "__main__":
    main()
