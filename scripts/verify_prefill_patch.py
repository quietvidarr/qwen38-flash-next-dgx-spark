#!/usr/bin/env python3
"""Verify the public patched source or installed wheel against measured B bytes.

No CUDA, Torch, weights or model loading. Runtime source hashes are from the
screened B integration. The CUDA test retains only the seven executed expert
cases; unrelated dense tests were removed. Wheel packaging is a separate fix.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

EXPECTED = {'src/tensorfold/cuda/exl3/experts.cpp': 'd26f462977ad9775de6e462222705b80c8ecaea8846899fd05251c09d6fc71e3', 'src/tensorfold/cuda/exl3/experts.cu': '4b025a89e8641521d45f2466674be513c8eedca5d001b25abe92f249f3ad5ee5', 'src/tensorfold/cuda/exl3/experts.py': '227851fef8356b5d104047e3b00adbaf956432eeda5f1002b9fca8b1d00b14bf', 'src/tensorfold/cuda/exl3/experts_cb0.cu': '4334d89f2314ebe4752362aefbf93f5109805562b3c0967b81ba2c053470f540', 'src/tensorfold/cuda/exl3/experts_cb1.cu': 'daa439de6b513fbc3370da78b4d84b3dcd7174eef27a463dd96620e4ce526b8d', 'src/tensorfold/cuda/exl3/experts_cb2.cu': '37ad3a30aed82ceced3e766be3604905388d19043a7635373340445e52fe323d', 'src/tensorfold/cuda/exl3/experts_prompt.cuh': 'e87f41d26d4de0f43ce2407462d2e96c5636d8c4e43b210197ee65dd8da4ada6', 'src/tensorfold/cuda/exl3/kernel212.py': '1b1bc620353c715a006601768f0d6fcf90277d1551f65e6d0d04a910a8f9be06', 'src/tensorfold/cuda/exl3/legacy212/experts.cpp': 'b667ff9b7dabc2b2bc5c2bf77c762ebf3acd7e89e90218e2cd370772a21348e5', 'src/tensorfold/cuda/exl3/legacy212/experts.cu': '807efd8f241edabce41cd2e149fbf8d61ea6875ca984de2ddf0411025c09a4b5', 'src/tensorfold/cuda/exl3/legacy212/experts_cb0.cu': '719ce51b1eb0218a36578adb9145358ee1271f0bbc72cfc575f3a273856bc236', 'src/tensorfold/cuda/exl3/legacy212/experts_cb1.cu': 'dad1e479d6047d4af6b30d38dab492df4534b829f8ef34d69552f6f033ee8825', 'src/tensorfold/cuda/exl3/legacy212/experts_cb2.cu': 'bcf5ea9c9d6f36dafa3132f52529006dfe0853425bad03040a9e37164f9266c4', 'src/tensorfold/cuda/exl3/legacy212/experts_grouped.cuh': 'cb2a016df23375db577ca2d9199ba145eb176e71a22295bdf386caba1cebff22', 'src/tensorfold/families/qwen4_exp/cuda/exl3_mm.py': 'aadf0af3e8b56441a1d4bd2d2eccc406906d9a98a8412ba85999268d1e63878a', 'src/tensorfold/families/qwen4_exp/cuda/exl3_pack.py': '660ba17f4df1f81838bbe7140e1b18b590b568b39521befecf81cf56fb633410', 'src/tensorfold/families/qwen4_exp/cuda/forward.py': '5af379862d4703c5a56e466f218c2b9d0f12c515899b377ef172540b8cc53b40', 'tests/cuda/test_exl3_prompt_experts.py': '3e63646f71fde1646497f41ec5f6e95a56abe54e9597bb3426f268faa97573e7'}


def load_policy(root, enabled, installed=False):
    relative = "tensorfold/cuda/exl3/kernel212.py" if installed else "src/tensorfold/cuda/exl3/kernel212.py"
    spec = importlib.util.spec_from_file_location("prefill_policy_witness", root / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"QWEN_PREFILL_KERNEL": enabled}):
        spec.loader.exec_module(module)
    return module


def verify(root, installed=False):
    checked = 0
    for relative, expected in EXPECTED.items():
        if installed and not relative.startswith("src/"):
            continue
        path = root / (relative.removeprefix("src/") if installed else relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise AssertionError("Missing or unmeasured source: " + relative)
        checked += 1
    off, on = load_policy(root, "0", installed), load_policy(root, "1", installed)
    assert not off.eligible(2048, prompt=True) and off.MOE_WINDOW == 1024
    assert on.MOE_WINDOW == 2048
    assert not on.eligible(64, prompt=True) and on.eligible(65, prompt=True)
    assert not on.eligible(2048, prompt=False) and not on.eligible(2048, prompt=True, group=False)
    assert all(not on.eligible(n, prompt=False) for n in range(1, 8))
    assert off.engagement()["prompt_dispatch_calls"] == 0
    on.record(65, 3)
    assert on.engagement()["prompt_pairs"] == 195
    assert on.admission_extra(2048, 11, 257) > 0
    try:
        load_policy(root, "invalid", installed)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid switch accepted")
    print(f"Verified {checked} controlled source hashes, legacy source closure and prompt/decode dispatch policy")
    return checked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--installed", action="store_true", help="Runtime is a site-packages root")
    args = p.parse_args()
    verify(args.runtime, args.installed)


if __name__ == "__main__":
    main()
