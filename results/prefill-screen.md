# Measured fresh-prefill screen

**B is the provisionally useful prompt-kernel integration. C is rejected** because all four short decode domains lost 6.06% to 7.76% versus A, beyond the frozen 3% screening margin. C is B plus selective read-ahead, not an independent control. The study ended with the incumbent restored and generation-verified.

One NVIDIA GB10 / DGX Spark. EXL3 4.05bpw_h6_ng6 language checkpoint, fixed revision `55a732e0c4c3d4614bc42b68493bb930d9b02c0a`; TensorFold base `609ca419abecebdc5a059498a613680bd3aa847f`. Greedy sampling, BF16 KV, explicit 262144 context, two stream slots, CUDA graphs, MTP depth6/confidence0.7. Every measured row reused zero prefix tokens.

Measured integration, not invention: original TensorFold PR212 prompt expert kernels by **grearjake-star**, head `f28b49f6e91b1789e25345875d61c5e663c19b1a`; PR252 selective n-gram read-ahead by **grearjake-star**, head `7ebfd533d72b827cd23661e307f8d4eaa0d1fb59`. PR212 unrelated F16/dense GEMM changes were excluded. Full source artifact hashes and source-receipt hashes are in [prefill-screen.json](prefill-screen.json), with source filesystem locations withheld.

Published B is opt-in via `--prefill-kernels`, default off. The public patch contains the 18 measured runtime/test files plus one **packaging-only** `pyproject.toml` fix for legacy212 source data. That fix is not a new GPU sweep. Measured B adds **480285400 bytes** of capacity-plan workspace; PR252/C is not shipped. Public patch SHA256: `bf884015181a1241b4038b25fdafff0f7d4887ce5725f112d99ba2f460f309fa`.

The existing [32k/64k before/after PNG](prefill-screen.png) is recorded by actual SHA256, byte size and dimensions in the JSON. Its rounded values represent only cold-advice-unverified A/B rows, not warm rows or C.

## All prefill cells

Each A/B/C value is **first-sample-inclusive prompt tok/s / TTFT seconds**. Changes compare the rate with A. JSON retains native prefill seconds, request wall seconds, output/count/hash evidence, faults, draft counters and treatment engagement for every cell.

| Actual prompt / domain | Page state | A rate / TTFT | B rate / TTFT | C rate / TTFT | B vs A | C vs A |
|---|---|---:|---:|---:|---:|---:|
| 8192 code | cold-advice-unverified | 373.80 / 21.958 | 526.65 / 15.567 | 1039.13 / 7.896 | +40.89% | +177.99% |
| 8192 code | warm, prefix-uncached | 731.40 / 11.605 | 1724.01 / 4.941 | 1719.27 / 4.974 | +135.71% | +135.07% |
| 32020 code | cold-advice-unverified | 557.34 / 57.817 | 1036.33 / 31.086 | 1618.88 / 19.968 | +85.94% | +190.47% |
| 32020 code | warm, prefix-uncached | 700.52 / 46.073 | 1627.69 / 19.859 | 1621.97 / 19.934 | +132.36% | +131.54% |
| 32196 prose | cold-advice-unverified | 411.37 / 78.647 | 616.53 / 52.406 | 1354.55 / 23.960 | +49.87% | +229.28% |
| 32196 prose | warm, prefix-uncached | 718.61 / 45.187 | 1644.73 / 19.759 | 1641.25 / 19.802 | +128.88% | +128.39% |
| 65536 code | cold-advice-unverified | 668.41 / 98.504 | 1571.53 / 41.926 | 1567.36 / 42.036 | +135.12% | +134.49% |
| 65536 code | warm, prefix-uncached | 668.53 / 98.484 | 1570.53 / 41.953 | 1568.79 / 41.999 | +134.92% | +134.66% |
| 65536 prose | cold-advice-unverified | 676.97 / 97.266 | 1579.07 / 41.726 | 1576.04 / 41.811 | +133.26% | +132.81% |
| 65536 prose | warm, prefix-uncached | 676.78 / 97.300 | 1577.17 / 41.780 | 1576.70 / 41.793 | +133.04% | +132.97% |

## All decode cells

Post-first-emission-chunk tok/s, EOS included. The first chunk is subtracted in full. All arms have matching emitted counts and finish reasons. `n/a` is not a throughput estimate.

| Actual prompt / domain | Cap / emitted / finish | A tok/s | B tok/s | C tok/s | B vs A | C vs A |
|---|---|---:|---:|---:|---:|---:|
| 44 speed/code | 400 / 400 / length | 48.546 | 48.611 | 45.602 | +0.13% | -6.06% |
| 76 speed/math | 400 / 400 / length | 68.088 | 68.101 | 62.803 | +0.02% | -7.76% |
| 48 speed/prose | 400 / 400 / length | 50.378 | 50.152 | 46.883 | -0.45% | -6.94% |
| 57 speed/structured | 400 / 400 / length | 73.857 | 73.509 | 69.245 | -0.47% | -6.25% |
| 32020 prefill/32000-code | 400 / 1 / stop | n/a | n/a | n/a | n/a | n/a |
| 65536 prefill/65536-prose | 400 / 400 / length | 54.742 | 54.649 | 53.144 | -0.17% | -2.92% |

## Bounded correctness, preparation and safety

- **26/26 A/B and 26/26 A/C emitted-token sequences match**, including EOS. Each arm retains all 10 prefill, 6 decode and 10 witness rows, plus 9 passed serving smokes.
- **Five serial/drafted witness pairs match in each arm**: code, math, prose, structured output and nominal 32k code. The last witness emits one EOS, not a sustained decode sample.
- **31 combined CPU tests + 17 collector CPU tests + 7 native targeted GPU tests passed.** CPU mocks, CUDA kernel tests and full-model measured rows are separate evidence levels.
- Reserve minima: **26.89 GiB** across recorded CPU preparation boundaries, **28.81 GiB** in the continuously supervised block. Frozen floor 7 GiB; host swap growth limit 1 GiB. One fixed block baseline, never reset between arms. Maximum host swap growth **0 bytes**; maximum owned-cgroup swap **0 bytes**, swap.max=0. No hard safety abort.
- Original service definition, image, source and backend settings were restored. The final retained route check confirms a stopped experiment, a completed generation with stop finish, default thinking and image-prefix-cache behavior. This exporter did not run a new live endpoint probe.

## Limits and interpretation

One pass per condition in fixed A/B/C order. B's measured decode changes are within the 3% practical margin; this is **not statistical equivalence**. Cold-advice-unverified is file-scoped clean-page advice, not independently verified cold residency. Warm means immediately repeated prompt with prefix state cleared, not a resident whole table. Both native prefill rate and prompt-tokens/TTFT are first-sample/emission proxies, not isolated GPU prefill throughput.

Nominal 32k prompts are actually 32020 code / 32196 prose tokens. The 32k-code decode cell emits **one EOS at cap400**, so there is no decode estimate. Four short domains reach cap400 with length finish; these are timing probes, **not finished-answer quality proof**. Historical short fixtures use their retained no-thinking rendering; serving default-thinking is tested separately.

Matched bounded token IDs do not establish universal losslessness or answer quality. Python kernel dispatch counters do not measure CUDA completion/graph replay; completed read-ahead advice does not prove residency. C's short decode loss is retained without inventing a cause. JSON contains hashes rather than prompt/output/reasoning text or raw token arrays; no raw study media, host identifiers, source locations or endpoints are exported.

## Reproduce the export

```sh
python scripts/export_prefill_results.py
python scripts/export_prefill_results.py --study STUDY_DIRECTORY --check
python scripts/export_prefill_results.py --study STUDY_DIRECTORY
```

The first command validates the committed JSON and deterministic report. `--study` is optional and supplied by the operator; `--check` compares against the retained source without writing. Regeneration writes only these two result files. No dependencies, GPU access, model loading, Git commits or pushes are performed.
