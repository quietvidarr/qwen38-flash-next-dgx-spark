#!/usr/bin/env python3
"""Verify the HF-cache copies of the weights against upstream's manifests (manifests/*.json): every listed file
present with its size and digest (sha256, or git-blob sha1 for small git files). Unlike upstream's
scripts/download_model.py --verify-only it follows the cache's symlinks.
  local/verify_weights.py          full check (reads ~100 GB, a few minutes)
  local/verify_weights.py --quick  sizes only"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))


def check(manifest: str, only: set[str] | None, quick: bool) -> int:
    m = json.loads((ROOT / "manifests" / manifest).read_text())
    snap = HUB / f"models--{m['repo'].replace('/', '--')}" / "snapshots" / m["revision"]
    bad = 0
    for item in m["files"]:
        if only and item["path"] not in only:
            continue
        p = snap / item["path"]
        if not p.is_file():
            print(f"MISSING {manifest}: {item['path']}"); bad += 1; continue
        if p.stat().st_size != item["bytes"]:
            print(f"SIZE    {manifest}: {item['path']} {p.stat().st_size} != {item['bytes']}"); bad += 1; continue
        if quick:
            continue
        h = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
        if item["algorithm"] == "git-blob-sha1":
            h.update(f"blob {item['bytes']}\0".encode())
        with open(p, "rb") as f:
            while chunk := f.read(16 << 20):
                h.update(chunk)
        if h.hexdigest() != item["digest"]:
            print(f"DIGEST  {manifest}: {item['path']}"); bad += 1
    print(f"{manifest}: {'OK' if not bad else f'{bad} problem(s)'} ({snap})")
    return bad


quick = "--quick" in sys.argv
bad = check("exl3-405.json", None, quick) + check("vision-bf16.json", None, quick)
sys.exit(1 if bad else 0)
