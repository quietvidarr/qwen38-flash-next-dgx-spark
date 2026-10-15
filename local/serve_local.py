#!/usr/bin/env python3
"""Upstream's serve.py with our engine options and model id, taken from the environment (local/config.sh).

serve.main() looks up engine_options() and MODEL_ID as module globals at call time, so replacing them here
changes what it serves without editing upstream's file. local/check-update.sh checks that this still holds."""
import os
import sys

sys.path.insert(0, "/opt/recipe")
import serve  # noqa: E402  upstream's scripts/serve.py, copied into the image by its Dockerfile

_upstream_options = serve.engine_options


def engine_options():
    o = dict(_upstream_options())
    o["streams"] = int(os.environ.get("STREAMS", o["streams"]))
    o["kv_dtype"] = os.environ.get("KV_DTYPE", o["kv_dtype"])
    o["prefetch"] = os.environ.get("PREFETCH", "1" if o["prefetch"] else "0") == "1"
    return o


serve.engine_options = engine_options


def _defaults(sampling):
    """Fill the server's default sampling from DEFAULT_* (upstream: greedy, top_k off). They apply only to fields a
    request omits. top_k matters for speed: a sampled request with top_k off draws through
    tensorfold/cuda/sampling.py nucleus_rows, a numpy sort on one CPU thread per drafted token (~10 tok/s)."""
    out = dict(sampling or {})
    for key, cast in (("temperature", float), ("top_k", int), ("top_p", float), ("min_p", float)):
        v = os.environ.get(f"DEFAULT_{key.upper()}", "")
        if v != "":
            out[key] = cast(v)
    return out


import tensorfold.cuda.server as _server  # noqa: E402  serve.main() imports App from here at call time

_App = _server.App


class App(_App):
    def __init__(self, *args, sampling=None, **kwargs):
        super().__init__(*args, sampling=_defaults(sampling), **kwargs)


_server.App = App


def _profiler(trigger="/cache/profile.trigger"):
    """Opt-in (TF_PROFILE=1): touch /cache/profile.trigger (the recipe's runtime-cache/) and the next N seconds (the
    file's content, default 20) of CUDA kernels are summed into /cache/profile-<time>.txt. Off by default."""
    import threading
    import time

    def run():
        import torch
        from torch.profiler import ProfilerActivity, profile
        while True:
            time.sleep(1)
            if not os.path.exists(trigger):
                continue
            try:
                secs = float(open(trigger).read().strip() or 20)
            except ValueError:
                secs = 20.0
            os.remove(trigger)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                time.sleep(secs)
                torch.cuda.synchronize()
            out = f"/cache/profile-{time.strftime('%H%M%S')}.txt"
            with open(out, "w") as f:
                f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60, max_name_column_width=90))
            if os.environ.get("TF_PROFILE_TRACE") == "1":      # the kernel timeline too (GPU idle gaps)
                prof.export_chrome_trace(out[:-4] + ".json")
            print(f"[serve_local] profile written to {out}", flush=True)

    threading.Thread(target=run, daemon=True, name="tf-profile").start()


if os.environ.get("TF_PROFILE") == "1":
    _profiler()
serve.MODEL_ID = os.environ.get("SERVED_NAME", serve.MODEL_ID)
print(f"[serve_local] model id {serve.MODEL_ID}; engine options {engine_options()}; n-gram reads "
      f"{os.environ.get('TF_EXL3_NGRAM_READ', 'pread')} x{os.environ.get('TF_EXL3_NGRAM_READERS', '32')}; default sampling "
      f"{_defaults({'temperature': 0, 'top_k': 0, 'top_p': 1, 'min_p': 0})} (fields a request omits)", flush=True)
raise SystemExit(serve.main())
