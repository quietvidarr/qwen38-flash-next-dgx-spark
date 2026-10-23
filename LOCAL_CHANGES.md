# Local changes

Upstream: [sxuff/qwen38-flash-next-dgx-spark](https://github.com/sxuff/qwen38-flash-next-dgx-spark) (souf). TensorFold: [ashhart/TensorFold](https://github.com/ashhart/TensorFold).
TensorFold changes live in [quietvidarr/TensorFold-spark@4877e2ad](https://github.com/quietvidarr/TensorFold-spark/commit/4877e2adf4b49bf6f5118d992618e68df8866a3c) (`local/pins.env`).
Branch `local-patches` only adds files (`local/`, this file). Settings in `.env` (git-ignored).

## Update

```bash
local/check-update.sh   # read-only. 0 = ok, 3 = incompatible, 1 = check failed
local/update.sh         # re-checks; if ok: rebase + image build
```
Never plain `git pull`/`rebase`. The check runs in a scratch worktree: our commits rebase, contact points exist, the fork is based on upstream's TensorFold pin and carries upstream's TensorFold patches.
Incompatible: fix `local/`, or rebase the fork and pin the new commit, then re-check.

### Contact points in upstream

| Upstream file | Relied on | Used by |
|---|---|---|
| `scripts/serve.py` | `engine_options()` (`streams`, `prefetch`, `kv_dtype`), `MODEL_ID`, `main()` importing `App`, `sampling=` | `local/serve_local.py` |
| `docker/Dockerfile` | `COPY scripts/serve.py`, TensorFold at `/opt/TensorFold` pinned by sha, its patches (only `vision-prefix-cache.patch`, in the fork) | `local/Dockerfile`, `config.sh`, `check-update.sh` |
| `scripts/supervise.py` | `IMAGE`, `docker_command()` | `local/config.sh`, `start.sh` |
| `manifests/*.json` | `repo`, `revision`, `branch`, file size + digest | `config.sh`, `download.sh`, `verify_weights.py` |

## Paths (env, `.env`)

- `PROMPT_CACHE_DIR`: SSD prompt cache, default `~/.cache/tensorfold-prompt-cache`, shared with the 35B recipe; empty = off.
- `HF_HUB_CACHE`: weights, default `~/.cache/huggingface/hub`.
- `TENSORFOLD_SRC`: fork clone for `build.sh`, default `.cache/TensorFold-spark`. Public fork.

## Recipe (`local/`)

| File | Change |
|---|---|
| `config.sh` | Settings + upstream pins; `.env` overrides. Name `Qwen3.8-Flash-Next-EXL3-4.05bpw-TensorFold`, 127.0.0.1:8001 |
| `start.sh`, `stop.sh` | Named container, host network, `--restart no`, kept when stopped; upstream's 112 GiB cap, read-only root, non-root |
| `serve_local.py` | `STREAMS`, `KV_DTYPE`, `PREFETCH`, `SERVED_NAME`, `DEFAULT_*` sampling from env. `top_k` 20 default (0 = CPU nucleus sort, ~10 tok/s) |
| `Dockerfile`, `build.sh`, `tensorfold-src.sh`, `pins.env` | Upstream image with `/opt/TensorFold` replaced by the fork at `TENSORFOLD_FORK_REV`. Tag `qwen38-flash-next-tensorfold-local:<tag>-<hash>` |
| `download.sh`, `hf-download.sh`, `verify_weights.py` | Resumable parallel download (curl), verified against upstream manifests |
| `image.sh` | "Update available" = image for current recipe not built |
| `check-update.sh`, `update.sh` | Safe update, above |
| `templates/chat_template.jinja` | Stock template + sticky reasoning notes; mounted, `TF_CHAT_TEMPLATE` (`CHAT_TEMPLATE`, empty = stock). Needs a fork with `TF_CHAT_TEMPLATE` |
| `templates/test_sticky.py` | CPU check vs stock: `PYTHONPATH=<fork>/src python3 local/templates/test_sticky.py <snapshot dir>` |

### Sticky reasoning

Stock puts the effort instruction into the first system block: an effort switch re-prefills the whole conversation.
With `chat_template_kwargs` `reasoning_prefix` (level of the beginning) and `reasoning_changes` (`[[index, level]]`, a note before `messages[index]`), the override keeps the beginning and adds `<|im_start|>system\n<instruction><|im_end|>` notes: xhigh/low stock text, medium `Reasoning effort is set to medium.`, off `Thinking is off.`. Prefix off/medium: no instruction. Without the kwargs: stock, byte for byte.

## TensorFold changes

In the fork, on upstream's `609ca419` + `vision-prefix-cache.patch`. Numbers = the former patch files; details in its `CHANGES.md`.

| # | Change |
|---|---|
| 0001 | N-gram rows via parallel `pread` before the GPU wait (`TF_EXL3_NGRAM_READ`, `TF_EXL3_NGRAM_READERS`); table stays on SSD |
| 0002 | Prompt experts on a prefill kernel (was decode kernel, 57% of prefill). `TF_EXL3_PROMPT_ROWS` |
| 0003 | 0001 in C++ (`ngram_read.cpp`): 65k rows 390 → 150 ms SSD, 312 → 16 ms cached. `TF_EXL3_NGRAM_NATIVE=0` = Python |
| 0004 | Prompt kernel tuning, same results |
| 0005 | Each conversation keeps its prompt ends (`TF_FLASH_KEEP`). Reuse 0% → 98% (8 conv × 4 turns). Keep conversations ≤ `STREAMS` |
| 0006 | Prompt cache on disk (`PROMPT_CACHE_DIR`): 100 GiB / 72 h, 50 GiB pinned |
| 0007 | Expert grouping O(picks), 3-7x; bit-identical |
| 0008 | Prompt cache host copies on a side stream |
| 0009 | Decode graphs per slot size, kept across switches: 36/35/46 → 45/46/45 tok/s |
| 0010 | Resume point at every message start (`TF_POINTS_LAST=all`, `TF_POINTS_GAP`), never dropped (`keep_rank.py`) |
| 0011-0013 | Prompt MoE v2, exact deterministic sum, prompt-path fusions |
| 0014 | Copy drafting gated by the MTP head (`TF_FLASH_COPY=0` = off) |
| 0015-0027 | Decode kernels: 32-row linear, hc_gemv/`TF_FLASH_HC12`, router tiles, MoE combine, n-gram reads. Prefill +11-23%, decode +10-35% at 4-8 streams |
| 0028 | Short prompt pieces use prompt kernels: exact resume (8/8, was 4-5/8) |
| 0029 | Prompt-cache points inside one prefill pass: no pass ends at a message start (cold 44-message prompt was +43%) |

Settings: `STREAMS=8`, `TF_FLASH_KEEP=16`, `TF_EXL3_PROMPT_ROWS=4096`, `TF_EXL3_NGRAM_READERS=128`, `TF_FLASH_HC12=1`, `DEFAULT_TOP_K=20`, `TF_POINTS_LAST=all`, `PROMPT_CACHE_DIR`.
Rollback: `TF_LOCAL_IMAGE=<older tag>` in `.env`.
Profile: `TF_PROFILE=1`, then `echo 20 > runtime-cache/profile.trigger`.

## New TensorFold

Rebase the fork's `spark` onto it, push, pin the new commit in `local/pins.env` (and the 35B recipe's).
Upstream's own test suite has ~18 failures in this image (no `mlx`, no models). Compare to unpatched, not zero.

## Measured (patches 0001-0005)

| | before | after | vLLM NVFP4 |
|---|---|---|---|
| prefill ~2k / 8k / 32k tok/s | — / ~650 / ~650 | 1,771 / 2,004 / 2,056 | 799 / 2,142 / 2,159 |
| decode, 1 request | 39-50 | 48-49 | 45-51 |
| 4 / 8 concurrent, total tok/s | — | 90 / 118 | 74 / 81 |
| follow-up cache reuse, 8 conv | 0% | 98% | — |

Prefill ~500-700 tok/s after an update = n-gram read path regressed; check logs for `n-gram reads pread`.
